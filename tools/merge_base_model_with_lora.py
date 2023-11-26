"""
Usage:
python merge_llama_with_chinese_lora.py \
    --base_model path/to/llama/model \
    --lora_model path/to/first/lora/model [path/to/second/lora/model] \
    --output_type [pth|huggingface] \
    --output_dir path/to/output/dir
"""
import os
import argparse
import torch
import peft
from peft import PeftModel
from llm.utils.tools.petrel_helper import PetrelHelper
from transformers import LlamaForCausalLM, LlamaTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")  # noqa
parser.add_argument('--output_dir', default='./', type=str)


emb_to_model_size = {
    4096: '7B',
    5120: '13B',
    6656: '30B',
    8192: '65B',
}
num_shards_of_models = {'7B': 1, '13B': 2}
params_of_models = {
    '7B':
        {
            "dim": 4096,
            "multiple_of": 256,
            "n_heads": 32,
            "n_layers": 32,
            "norm_eps": 1e-06,
            "vocab_size": -1,
        },
    '13B':
        {
            "dim": 5120,
            "multiple_of": 256,
            "n_heads": 40,
            "n_layers": 40,
            "norm_eps": 1e-06,
            "vocab_size": -1,
        },
}

WEIGHTS_NAME = "pytorch_model.bin"


def get_ceph_path(path):
    ceph_bucket = os.environ.get('CEPHBUCKET')
    if ceph_bucket[-1] != '/':
        ceph_bucket += '/'
    # remove /
    if path[0] == '/':
        path = path[1:]
    # remove ./
    if path[:2] == './':
        path = path[2:]
    ceph_path = ceph_bucket + path
    return ceph_path


def ceph_save(state_dict, path):
    ceph_path = get_ceph_path(path)
    with open(path + '.ceph', 'w') as f:
        print(ceph_path, file=f)
    PetrelHelper.save(state_dict, ceph_path)


if __name__ == '__main__':

    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip()) != 0]
    output_dir = args.output_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s) {lora_model_paths}:")

    # Original method without offloading
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    # infer the model size from the checkpoint
    embedding_size = base_model.get_input_embeddings().weight.size(1)
    model_size = emb_to_model_size[embedding_size]
    print(f"Peft version: {peft.__version__}")
    print(f"Loading LoRA for {model_size} model")

    lora_model = None
    lora_model_sd = None
    for lora_index, lora_model_path in enumerate(lora_model_paths):
        print(f"Loading LoRA {lora_model_path}")
        tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
        if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Extended vocabulary size to {len(tokenizer)}")

        first_weight = base_model.model.layers[0].self_attn.q_proj.weight
        first_weight_old = first_weight.clone()

        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            device_map={"": "cpu"},
            torch_dtype=torch.float16,
        )
        assert torch.allclose(first_weight_old, first_weight)
        print("Merging with merge_and_unload...")
        base_model = lora_model.merge_and_unload()
    tokenizer.save_pretrained(output_dir)
    print("Saving to Hugging Face format...")
    if 'CEPHBUCKET' in os.environ and os.environ.get('CEPHBUCKET') is not None:
        save_function = ceph_save
    else:
        save_function = torch.save
    LlamaForCausalLM.save_pretrained(base_model, output_dir, save_function=save_function)

    if os.environ.get('CEPHBUCKET', None) is not None:
        all_files = os.listdir(output_dir)
        for file_path in all_files:
            if file_path.endswith('.bin'):
                continue
            local_path = os.path.join(output_dir, file_path)
            ceph_file_path = get_ceph_path(local_path)
            with open(local_path, 'rb') as f:
                PetrelHelper.write(f, ceph_file_path)
