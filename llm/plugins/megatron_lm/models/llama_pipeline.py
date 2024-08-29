from .utils import (
    LayerSpec,
    partition_uniform,
    partition_balanced
)

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEDelayedScaling,
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None
    try:
        import apex

        LayerNormImpl = FusedLayerNorm
    except ModuleNotFoundError:
        from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

        LayerNormImpl = WrappedTorchLayerNorm


class LlaMAModelPipe(GPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # define specs list
        self.specs = self.build_specs()
        # initialize partition
        self._partition_layers()

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder
        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # model build
        self.forward_funcs = []
        self.build()

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()
        
        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )

    def build(self):
        specs = self.specs

        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start

            if isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.add_module(name, module)
            else:
                raise NotImplementedError('Only LayerSpec was supported.')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def _partition_layers(self):
        num_stages = mpu.get_pipeline_model_parallel_world_size()
        stage_id = mpu.get_pipeline_model_parallel_rank()

        args = gets_args()
        method = args.pp_partition_method
        method = method.lower()

        if method == "uniform":
            num_layers = len(self.specs)
            self.parts = partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == "parameters":
            param_counts = self._count_layer_params()
            self.parts = partition_balanced(weights=param_counts, num_parts=num_stages)
        elif "manual" in method:
            parts = method.split("manual:")[1].split(',')
            self.parts = [int(item) for item in parts]
        elif method.startswith('type:'):
            # TODO
            pass
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self.specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
        
        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self.specs)
        for idx, layer in enumerate(self.specs):
            if isinstance(layer, LayerSpec):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def build_specs(self):
        specs = []

        # word embedding
        word_embedings_params = dict(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
        )
        specs.append(LayerSpec(LanguageModelEmbedding, **word_embedings_params))

        # transformer layer
        transformer_layer_params = dict(
            config=self.config
        )
        if hasattr(self.transformer_layer_spec, "submodules") and self.transformer_layer_spec.submodules is not None:
            transformer_layer_params["submodules"] = self.transformer_layer_spec.submodules
        if hasattr(self.transformer_layer_spec, "params"):
            transformer_layer_params["extra_params"] = self.transformer_layer_spec.params
        for layer_idx in range(self.config.num_layers):
            transformer_layer_params["layer_number"] = layer_idx + 1
            specs.append(LayerSpec(TransformerLayer, **transformer_layer_params))

        # final layernorm after transformer layers
        layer_norm_params = dict(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon
        )
        specs.append(LayerSpec(LayerNormImpl, **layer_norm_params))

        # lm_head
        if self.config.defer_embedding_wgrad_compute:
            # The embedding activation buffer preserves a reference to the input activations
            # of the final embedding projection layer GEMM. It will hold the activations for
            # all the micro-batches of a global batch for the last pipeline stage. Once we are
            # done with all the back props for all the microbatches for the last pipeline stage,
            # it will be in the pipeline flush stage. During this pipeline flush we use the
            # input activations stored in embedding activation buffer and gradient outputs stored
            # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None
        lm_head_params = dict(
            input_size=self.config.hidden_size,
            output_size=self.vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=False, # self.pre_process and self.share_embeddings_and_output_weights,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )
        specs.append(LayerSpec(tensor_parallel.ColumnParallelLinear, **lm_head_params))

        return specs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        rotary_pos_emb = None
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = input_ids
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        if repr(self.specs[self._local_start]) == "TransformerLayer":
            if self.position_embedding_type == 'rope':
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.decoder, decoder_input, self.config
                )
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        
        for idx in range(len(self.forward_funcs)):
            if self.pre_process and idx == 0:
                # embeddings
                decoder_input = self.forward_funcs[idx](input_ids=decoder_input, position_ids=position_ids)
            elif repr(self.specs[self._local_start + idx]) == "TransformerLayer":
                decoder_input, context = self.forward_funcs[idx](
                    hidden_states=decoder_input,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                )
            elif repr(self.specs[self._local_start + idx]) == "LayerNormImpl":
                decoder_input = self.forward_funcs[idx](decoder_input)
            elif repr(self.specs[self._local_start + idx]) == "tensor_parallel.ColumnParallelLinear":
                # logits and loss
                output_weight = None
                if self.share_embeddings_and_output_weights:
                    output_weight = self.shared_embedding_or_output_weight()
                logits, _ = self.output_layer(decoder_input, weight=output_weight)

                if has_config_logger_enabled(self.config):
                    payload = OrderedDict(
                        {
                            'input_ids': input_ids,
                            'position_ids': position_ids,
                            'attention_mask': attention_mask,
                            'decoder_input': decoder_input,
                            'logits': logits,
                        }
                    )
                    log_config_to_disk(self.config, payload, prefix='input_and_logits')

                if labels is None:
                    # [s b h] => [b s h]
                    # return logits.transpose(0, 1).contiguous()
                    decoder_input = logits.transpose(0, 1).contiguous()
                else:
                    # loss
                    decoder_input = self.compute_language_model_loss(labels, logits)

        return decoder_input