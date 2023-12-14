FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y \
	zsh \
	tmux \
	git \
	vim \
	build-essential \
        libibverbs-dev

# easyllm 1.0.0
RUN pip install https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=a7a49d459bf4862f64f7bc1a68beccf8881c2fa9f3e0569608e16ba6f85ebf7b # http://spring.sensetime.com/pypi/packages/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl

RUN pip install py-cpuinfo==9.0.0 \
        hjson==3.1.0 \
        pydantic==1.10.8 \
        safetensors==0.3.1 \
        tokenizers==0.14.1 \
        accelerate==0.20.3 \
        pandas==1.5.2 \
        einops==0.6.1 \
        sentencepiece==0.1.99 \
        tiktoken==0.5.1

RUN pip install huggingface-hub==0.16.4 \
        peft==0.4.0 \
        transformers==4.34.0 \
        http://spring.sensetime.com/pypi/packages/flash_attn-2.2.5+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
        http://spring.sensetime.com/pypi/packages/dropout_layer_norm-0.1-cp310-cp310-linux_x86_64.whl
