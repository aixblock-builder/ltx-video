from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Dict, Optional
from logging_class import write_log, start_queue
import constants as const
from vms.config import DEFAULT_CAPTION_DROPOUT_P, DEFAULT_DATASET_TYPE, DEFAULT_MIXED_PRECISION, DEFAULT_PROMPT_PREFIX, DEFAULT_REMOVE_COMMON_LLM_CAPTION_PREFIXES, DEFAULT_RESHAPE_MODE, DEFAULT_SEED, HF_API_TOKEN, MD_TRAINING_BUCKETS, RESOLUTION_OPTIONS, SD_TRAINING_BUCKETS, TrainingConfig, DEFAULT_TRAINING_TYPE, DEFAULT_NUM_GPUS, DEFAULT_PRECOMPUTATION_ITEMS, DEFAULT_NB_LR_WARMUP_STEPS
from vms.utils.finetrainers_utils import prepare_finetrainers_dataset
from vms.utils.gpu_detector import get_available_gpu_count
import os
import psutil


channel_log = const.CHANNEL_LOGS
log_queue, _ = start_queue(channel_log)
write_log(log_queue)
output_path = const.OUTPUTS_DIR
output_session_file = output_path / "session.json"
training_path = const.PROJ_DIR / "training"
training_videos_path = training_path / "videos"

def start_training(
    lora_rank: str,
    lora_alpha: str,
    train_steps: int,
    batch_size: int, 
    learning_rate: float,
    save_iterations: int,
    repo_id: str,
    model_type: str = "ltx_video",
    resolution: str = list(RESOLUTION_OPTIONS.keys())[0],
    training_type: str = DEFAULT_TRAINING_TYPE,
    model_version: str = "",
    resume_from_checkpoint: Optional[str] = None,
    num_gpus: int = DEFAULT_NUM_GPUS,
    precomputation_items: int = DEFAULT_PRECOMPUTATION_ITEMS,
    lr_warmup_steps: int = DEFAULT_NB_LR_WARMUP_STEPS,
    custom_prompt_prefix: Optional[str] = None,
):
    accelerate_config = None
    if num_gpus == 1:
        accelerate_config = "accelerate_configs/uncompiled_1.yaml"
    elif num_gpus == 2:
        accelerate_config = "accelerate_configs/uncompiled_2.yaml"
    elif num_gpus == 4:
        accelerate_config = "accelerate_configs/uncompiled_4.yaml"
    elif num_gpus == 8:
        accelerate_config = "accelerate_configs/uncompiled_8.yaml"
    else:
        # Default to 1 GPU config if no matching config is found
        accelerate_config = "accelerate_configs/uncompiled_1.yaml"
        num_gpus = 1
        visible_devices = "0"

    resolution_option = resolution
    training_buckets_name = RESOLUTION_OPTIONS.get(resolution_option, "SD_TRAINING_BUCKETS")
    
    # Determine which buckets to use based on the selected resolution
    if training_buckets_name == "SD_TRAINING_BUCKETS":
        training_buckets = SD_TRAINING_BUCKETS
    elif training_buckets_name == "MD_TRAINING_BUCKETS":
        training_buckets = MD_TRAINING_BUCKETS
    else:
        training_buckets = SD_TRAINING_BUCKETS
    
    current_dir = Path(__file__).parent
    train_script = current_dir / "train.py"

    videos_file, prompts_file = prepare_finetrainers_dataset()
    if videos_file is None or prompts_file is None:
        error_msg = "Failed to generate training lists"
        print(error_msg)
        return error_msg, "Training preparation failed"

    video_count = sum(1 for _ in open(videos_file))
    print(f"Generated training lists with {video_count} files")

    if video_count == 0:
        error_msg = "No training files found"
        print(error_msg)
        return error_msg, "No training data available"

    # Use different launch commands based on model type
    # For Wan models, use torchrun instead of accelerate launch
    if model_type == "wan":
        # Configure torchrun parameters
        torchrun_args = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=" + str(num_gpus),
            "--nnodes=1",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(train_script)
        ]
        
        # Additional args needed for torchrun
        config_args.extend([
            "--parallel_backend", "ptd",
            "--pp_degree", "1", 
            "--dp_degree", "1", 
            "--dp_shards", "1", 
            "--cp_degree", "1", 
            "--tp_degree", "1"
        ])
        
        # Log the full command for debugging
        command_str = ' '.join(torchrun_args + config_args)
        print(f"Executing command: {command_str}")
        
        launch_args = torchrun_args
    else:
        # For other models, use accelerate launch as before
        # Determine the appropriate accelerate config file based on num_gpus
        accelerate_config = None
        if num_gpus == 1:
            accelerate_config = "accelerate_configs/uncompiled_1.yaml"
        elif num_gpus == 2:
            accelerate_config = "accelerate_configs/uncompiled_2.yaml"
        elif num_gpus == 4:
            accelerate_config = "accelerate_configs/uncompiled_4.yaml"
        elif num_gpus == 8:
            accelerate_config = "accelerate_configs/uncompiled_8.yaml"
        else:
            # Default to 1 GPU config if no matching config is found
            accelerate_config = "accelerate_configs/uncompiled_1.yaml"
            num_gpus = 1
            visible_devices = "0"

        # Configure accelerate parameters
        accelerate_args = [
            "accelerate", "launch",
            "--config_file", accelerate_config,
            "--gpu_ids", visible_devices,
            "--mixed_precision=bf16",
            "--num_processes=" + str(num_gpus),
            "--num_machines=1",
            "--dynamo_backend=no",
            str(train_script)
        ]
        
        # Log the full command for debugging
        command_str = ' '.join(accelerate_args + config_args)
        print(f"Executing command: {command_str}")
        
        launch_args = accelerate_args

    if model_type == "hunyuan_video":
        flow_weighting_scheme = "none"
    else:
        flow_weighting_scheme = "logit_normal"

    if custom_prompt_prefix:
        custom_prompt_prefix = custom_prompt_prefix.rstrip(', ')

    # Create a proper dataset configuration JSON file
    dataset_config_file = output_path / "dataset_config.json"

    # Determine appropriate ID token based on model type and custom prefix
    id_token = custom_prompt_prefix  # Use custom prefix as the primary id_token

    # Only use default ID tokens if no custom prefix is provided
    if not id_token:
        id_token = DEFAULT_PROMPT_PREFIX

    dataset_config = {
        "datasets": [
            {
                "data_root": str(training_path),
                "dataset_type": DEFAULT_DATASET_TYPE,
                "id_token": id_token,
                "video_resolution_buckets": [[f, h, w] for f, h, w in training_buckets],
                "reshape_mode": DEFAULT_RESHAPE_MODE,
                "remove_common_llm_caption_prefixes": DEFAULT_REMOVE_COMMON_LLM_CAPTION_PREFIXES,
            }
        ]
    }

    # Write the dataset config to file
    with open(dataset_config_file, 'w') as f:
        json.dump(dataset_config, f, indent=2)

    print(f"Created dataset configuration file at {dataset_config_file}")

    # Get config for selected model type with preset buckets
    if model_type == "hunyuan_video":
        if training_type == "lora":
            config = TrainingConfig.hunyuan_video_lora(
                data_path=str(training_path),
                output_path=str(output_path),
                buckets=training_buckets
            )
        else:
            # Hunyuan doesn't support full finetune in our UI yet
            error_msg = "Full finetune is not supported for Hunyuan Video due to memory limitations"
            print(error_msg)
            return error_msg, "Training configuration error"
    elif model_type == "ltx_video":
        if training_type == "lora":
            config = TrainingConfig.ltx_video_lora(
                data_path=str(training_path),
                output_path=str(output_path),
                buckets=training_buckets
            )
        else:
            config = TrainingConfig.ltx_video_full_finetune(
                data_path=str(training_path),
                output_path=str(output_path),
                buckets=training_buckets
            )
    elif model_type == "wan":
        if training_type == "lora":
            config = TrainingConfig.wan_lora(
                data_path=str(training_path),
                output_path=str(output_path),
                buckets=training_buckets
            )
        else:
            error_msg = "Full finetune for Wan is not yet supported in this UI"
            print(error_msg)
            return error_msg, "Training configuration error"
    config.train_steps = int(train_steps)
    config.batch_size = int(batch_size)
    config.lr = float(learning_rate)
    config.checkpointing_steps = int(save_iterations)
    config.training_type = training_type
    config.flow_weighting_scheme = flow_weighting_scheme
    
    config.lr_warmup_steps = int(lr_warmup_steps)
    num_gpus = min(num_gpus, get_available_gpu_count())
    if num_gpus <= 0:
        num_gpus = 1
    visible_devices = ",".join([str(i) for i in range(num_gpus)])
    config.data_root = str(dataset_config_file)
    config.lora_rank = int(lora_rank)
    config.lora_alpha = int(lora_alpha)
    # Common settings for both models
    config.mixed_precision = DEFAULT_MIXED_PRECISION
    config.seed = DEFAULT_SEED
    config.gradient_checkpointing = True
    config.enable_slicing = True
    config.enable_tiling = True
    config.caption_dropout_p = DEFAULT_CAPTION_DROPOUT_P
    config.precomputation_items = precomputation_items

    config_args = config.to_args_list()

    command_str = ' '.join(accelerate_args + config_args)
    print(f"Executing command: {command_str}")
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    env["WANDB_MODE"] = "offline"
    env["HF_API_TOKEN"] = HF_API_TOKEN
    env["FINETRAINERS_LOG_LEVEL"] = "DEBUG"  # Added for better debugging
    env["CUDA_VISIBLE_DEVICES"] = visible_devices

    process = subprocess.Popen(
        launch_args + config_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
        cwd=str(current_dir),
        bufsize=1,
        universal_newlines=True
    )


def validate_training_config(config: TrainingConfig, model_type: str) -> Optional[str]:
    """Validate training configuration"""
    print(f"Validating config for {model_type}")
    
    try:
        # Basic validation
        if not config.output_dir:
            return "Output directory not specified"
            
        # For the dataset_config validation, we now expect it to be a JSON file
        dataset_config_path = Path(config.data_root)
        if not dataset_config_path.exists():
            return f"Dataset config file does not exist: {dataset_config_path}"
        
        # Check the JSON file is valid
        try:
            with open(dataset_config_path, 'r') as f:
                dataset_json = json.load(f)
            
            # Basic validation of the JSON structure
            if "datasets" not in dataset_json or not isinstance(dataset_json["datasets"], list) or len(dataset_json["datasets"]) == 0:
                return "Invalid dataset config JSON: missing or empty 'datasets' array"
                
        except json.JSONDecodeError:
            return f"Invalid JSON in dataset config file: {dataset_config_path}"
        except Exception as e:
            return f"Error reading dataset config file: {str(e)}"
                
        # Check training videos directory exists
        if not training_videos_path.exists():
            return f"Training videos directory does not exist: {training_videos_path}"
            
        # Validate file counts
        video_count = len(list(training_videos_path.glob('*.mp4')))
        
        if video_count == 0:
            return "No training files found"
                
        # Model-specific validation
        if config.batch_size > 4:
                return "LTX model recommended batch size is 1-4"
                
        print(f"Config validation passed with {video_count} training files")
        return None
        
    except Exception as e:
        print(f"Error during config validation: {str(e)}")
        return f"Configuration validation failed: {str(e)}"
