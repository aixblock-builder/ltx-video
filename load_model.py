import os

import torch
from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
from huggingface_hub import HfFolder

# ---------------------------------------------------------------------------
# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)

from huggingface_hub import login

hf_access_token = "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"
login(token=hf_access_token)


def _load():
    model_id = "black-forest-labs/FLUX.1-dev"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_nf4 = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        # device_map=device,
    )
    pipe_demo = FluxPipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )


_load()
