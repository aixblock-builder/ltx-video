import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "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")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN", "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")
WANDB_TOKEN = os.getenv("WANDB_TOKEN", "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
AXB_TOKEN = os.getenv("AXB_TOKEN", "ebcf0ceda01518700f41dfa234b6f4aaea0b57af")
REPO_ID = "tonyshark/flux"

HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 23456

PROJ_DIR = Path.cwd()
MODELS_DIR = PROJ_DIR.joinpath("models")
DATASETS_DIR = PROJ_DIR.joinpath("datasets")
OUTPUTS_DIR = PROJ_DIR.joinpath("outputs")

CLONE_DIR = DATASETS_DIR

FRAMEWORK = "huggingface"
TASK = "text-to-image"
WORLD_SIZE = 1
RANK = 0
CHANNEL_LOGS = "training_logs.txt"

# model, minimum
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DATASET = "diffusers/dog-example"
# OUTPUT_DIR = "./model"
JSON_TRAINING_ARGS = "training_args.json"
ARGS_IGNORE = ["lora", "dataset_id"]

# model, extra
GPU_MEMORY = 12
IMG_COL = "image"
PROMPT_COL = "text"
PROMPT = "default prompt"
RESOLUTION = 512
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 5e-6
MAX_TRAINING_STEP = 2
PUSH_TO_HUB = True

MEMORY_CONFIG = {
    # 8:  f"""
    #         --optimizer_type adafactor
    #         --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"
    #         --split_mode
    #         --network_args "train_blocks=single"
    #         --lr_scheduler constant_with_warmup
    #         --max_grad_norm 0.0
    #     """, # experimental
    12: f"""
                          --optimizer_type adafactor
                          --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"
                          --lr_scheduler constant_with_warmup
                          --max_grad_norm 0.0 
                      """,
    16: f"""  
                          --optimizer_type adafactor
                          --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False"
                          --lr_scheduler constant_with_warmup 
                          --max_grad_norm 0.0 
                        """,
    29: "--optimizer_type adamw8bit ",
}
