import os
import uuid
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import math

def parse_bool_env(env_value: Optional[str]) -> bool:
    """Parse environment variable string to boolean
    
    Handles various true/false string representations:
    - True: "true", "True", "TRUE", "1", etc
    - False: "false", "False", "FALSE", "0", "", None
    """
    if not env_value:
        return False
    return str(env_value).lower() in ('true', '1', 't', 'y', 'yes')

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ASK_USER_TO_DUPLICATE_SPACE = parse_bool_env(os.getenv("ASK_USER_TO_DUPLICATE_SPACE"))

# For large datasets that would be slow to display or download
USE_LARGE_DATASET = parse_bool_env(os.getenv("USE_LARGE_DATASET"))

# Base storage path
STORAGE_PATH = Path(os.environ.get('STORAGE_PATH', '.data'))

# ----------- Subdirectories for different data types -----------
# The following paths correspond to temporary files, before they we "commit" (re-copy) them to the current project's training/ directory
VIDEOS_TO_SPLIT_PATH = STORAGE_PATH / "videos_to_split"    # Raw uploaded/downloaded files
STAGING_PATH = STORAGE_PATH / "staging"                    # This is where files that are captioned or need captioning are waiting
# --------------------------------------------------------------

# On the production server we can afford to preload the big model
PRELOAD_CAPTIONING_MODEL = parse_bool_env(os.environ.get('PRELOAD_CAPTIONING_MODEL'))

CAPTIONING_MODEL = "lmms-lab/LLaVA-Video-7B-Qwen2"

DEFAULT_PROMPT_PREFIX = "In the style of TOK, "

# This is only use to debug things in local
USE_MOCK_CAPTIONING_MODEL = parse_bool_env(os.environ.get('USE_MOCK_CAPTIONING_MODEL'))

DEFAULT_CAPTIONING_BOT_INSTRUCTIONS = "Please write a full video description. Be synthetic and methodically list camera (close-up shot, medium-shot..), genre (music video, horror movie scene, video game footage, go pro footage, japanese anime, noir film, science-fiction, action movie, documentary..), characters (physical appearance, look, skin, facial features, haircut, clothing), scene (action, positions, movements), location (indoor, outdoor, place, building, country..), time and lighting (natural, golden hour, night time, LED lights, kelvin temperature etc), weather and climate (dusty, rainy, fog, haze, snowing..), era/settings."

def generate_model_project_id() -> str:
    """Generate a new UUID for a model project"""
    return str(uuid.uuid4())

def get_global_config_path() -> Path:
    """Get the path to the global config file"""
    return STORAGE_PATH / "config.json"

def load_global_config() -> dict:
    """Load the global configuration file
    
    Returns:
        Dict containing global configuration
    """
    config_path = get_global_config_path()
    if not config_path.exists():
        # Create default config if it doesn't exist
        default_config = {
            "latest_model_project_id": None
        }
        save_global_config(default_config)
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading global config: {e}")
        return {"latest_model_project_id": None}

def save_global_config(config: dict) -> bool:
    """Save the global configuration file
    
    Args:
        config: Dictionary containing configuration to save
        
    Returns:
        True if successful, False otherwise
    """
    config_path = get_global_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving global config: {e}")
        return False

def update_latest_project_id(project_id: str) -> bool:
    """Update the latest project ID in global config
    
    Args:
        project_id: The project ID to save
        
    Returns:
        True if successful, False otherwise
    """
    config = load_global_config()
    config["latest_model_project_id"] = project_id
    return save_global_config(config)

def get_project_paths(project_id: str) -> Tuple[Path, Path, Path, Path]:
    """Get paths for a specific project
    
    Args:
        project_id: The model project UUID
        
    Returns:
        Tuple of (training_path, training_videos_path, output_path, log_file_path)
    """
    project_base = STORAGE_PATH / "models" / project_id
    training_path = project_base / "training"
    training_videos_path = training_path / "videos"
    output_path = project_base / "output"
    log_file_path = output_path / "last_session.log"
    
    # Ensure directories exist
    training_path.mkdir(parents=True, exist_ok=True)
    training_videos_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return training_path, training_videos_path, output_path, log_file_path

def migrate_legacy_project() -> Optional[str]:
    """Migrate legacy project structure to new UUID-based structure
    
    Returns:
        New project UUID if migration was performed, None otherwise
    """
    legacy_training = STORAGE_PATH / "training"
    legacy_output = STORAGE_PATH / "output"
    
    # Check if legacy folders exist and contain data
    has_training_data = legacy_training.exists() and any(legacy_training.iterdir())
    has_output_data = legacy_output.exists() and any(legacy_output.iterdir())
    
    if not (has_training_data or has_output_data):
        return None
        
    # Generate new project ID and paths
    project_id = generate_model_project_id()
    training_path, training_videos_path, output_path, log_file_path = get_project_paths(project_id)
    
    # Migrate data if it exists
    if has_training_data:
        # Copy files instead of moving to prevent data loss
        for file in legacy_training.glob("*"):
            if file.is_file():
                shutil.copy2(file, training_path)
        
        # Copy videos subfolder if it exists
        legacy_videos = legacy_training / "videos"
        if legacy_videos.exists():
            for file in legacy_videos.glob("*"):
                if file.is_file():
                    shutil.copy2(file, training_videos_path)
    
    if has_output_data:
        for file in legacy_output.glob("*"):
            if file.is_file():
                shutil.copy2(file, output_path)
            elif file.is_dir():
                # For checkpoint directories
                target_dir = output_path / file.name
                target_dir.mkdir(exist_ok=True)
                for subfile in file.glob("*"):
                    if subfile.is_file():
                        shutil.copy2(subfile, target_dir)
    
    return project_id

# Create directories
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
VIDEOS_TO_SPLIT_PATH.mkdir(parents=True, exist_ok=True)
STAGING_PATH.mkdir(parents=True, exist_ok=True)

# Add at the end of the file, after the directory creation section
# This ensures models directory exists
MODELS_PATH = STORAGE_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# To secure public instances
VMS_ADMIN_PASSWORD = os.environ.get('VMS_ADMIN_PASSWORD', '')

# Image normalization settings
NORMALIZE_IMAGES_TO = os.environ.get('NORMALIZE_IMAGES_TO', 'png').lower()
if NORMALIZE_IMAGES_TO not in ['png', 'jpg']:
    raise ValueError("NORMALIZE_IMAGES_TO must be either 'png' or 'jpg'")
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', '97'))

MODEL_TYPES = {
    "LTX-Video": "ltx_video",
    "HunyuanVideo": "hunyuan_video", 
    "Wan": "wan"
}

# Training types
TRAINING_TYPES = {
    "LoRA Finetune": "lora",
    "Full Finetune": "full-finetune",
    "Control LoRA": "control-lora",
    "Control Full Finetune": "control-full-finetune"
}

# Model versions for each model type
MODEL_VERSIONS = {
    "wan": {
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
            "name": "Wan 2.1 T2V 1.3B (text-only, smaller)",
            "type": "text-to-video",
            "description": "Faster, smaller model (1.3B parameters)"
        },
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
            "name": "Wan 2.1 T2V 14B (text-only, larger)",
            "type": "text-to-video",
            "description": "Higher quality but slower (14B parameters)"
        },
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": {
            "name": "Wan 2.1 I2V 480p (image+text)",
            "type": "image-to-video",
            "description": "Image conditioning at 480p resolution"
        },
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": {
            "name": "Wan 2.1 I2V 720p (image+text)",
            "type": "image-to-video",
            "description": "Image conditioning at 720p resolution"
        },
        "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers": {
            "name": "Wan 2.1 FLF2V 720p (frame conditioning)",
            "type": "frame-to-video",
            "description": "Frame conditioning (first/last frame to video) at 720p resolution"
        }
    },
    "ltx_video": {
        "Lightricks/LTX-Video": {
            "name": "LTX Video (official)",
            "type": "text-to-video",
            "description": "Official LTX Video model"
        }
    },
    "hunyuan_video": {
        "hunyuanvideo-community/HunyuanVideo": {
            "name": "Hunyuan Video (official)",
            "type": "text-to-video",
            "description": "Official Hunyuan Video model"
        }
    }
}

DEFAULT_SEED = 42

DEFAULT_REMOVE_COMMON_LLM_CAPTION_PREFIXES = True

DEFAULT_DATASET_TYPE = "video"
DEFAULT_TRAINING_TYPE = "lora"

DEFAULT_RESHAPE_MODE = "bicubic"

DEFAULT_MIXED_PRECISION = "bf16"



DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS = 200

DEFAULT_LORA_RANK = 128
DEFAULT_LORA_RANK_STR = str(DEFAULT_LORA_RANK)

DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_ALPHA_STR = str(DEFAULT_LORA_ALPHA)

DEFAULT_CAPTION_DROPOUT_P = 0.05

DEFAULT_BATCH_SIZE = 1

DEFAULT_LEARNING_RATE = 3e-5

# GPU SETTINGS
DEFAULT_NUM_GPUS = 1
DEFAULT_MAX_GPUS = min(8, torch.cuda.device_count() if torch.cuda.is_available() else 1)
DEFAULT_PRECOMPUTATION_ITEMS = 512

DEFAULT_NB_TRAINING_STEPS = 1000

# For this value, it is recommended to use about 20 to 40% of the number of training steps
DEFAULT_NB_LR_WARMUP_STEPS = math.ceil(0.20 * DEFAULT_NB_TRAINING_STEPS)  # 20% of training steps

# Whether to automatically restart a training job after a server reboot or not
DEFAULT_AUTO_RESUME = False

# Control training defaults
DEFAULT_CONTROL_TYPE = "canny"
DEFAULT_TRAIN_QK_NORM = False
DEFAULT_FRAME_CONDITIONING_TYPE = "full"
DEFAULT_FRAME_CONDITIONING_INDEX = 0
DEFAULT_FRAME_CONDITIONING_CONCATENATE_MASK = False

# For validation
DEFAULT_VALIDATION_NB_STEPS = 50
DEFAULT_VALIDATION_HEIGHT = 512
DEFAULT_VALIDATION_WIDTH = 768
DEFAULT_VALIDATION_NB_FRAMES = 49
DEFAULT_VALIDATION_FRAMERATE = 8

# you should use resolutions that are powers of 8
# using a 16:9 ratio is also super-recommended

# SD
SD_16_9_W = 1024 # 8*128
SD_16_9_H = 576  # 8*72
SD_9_16_W = 576  # 8*72
SD_9_16_H = 1024 # 8*128

# MD (720p)
MD_16_9_W = 1280 # 8*160
MD_16_9_H = 720  # 8*90
MD_9_16_W = 720  # 8*90
MD_9_16_H = 1280 # 8*160

# HD (1080p)
HD_16_9_W = 1920 # 8*240
HD_16_9_H = 1080 # 8*135
HD_9_16_W = 1080 # 8*135
HD_9_16_H = 1920 # 8*240

# QHD (2K)
QHD_16_9_W = 2160 # 8*270
QHD_16_9_H = 1440 # 8*180
QHD_9_16_W = 1440 # 8*180
QHD_9_16_H = 2160 # 8*270

# UHD (4K)
UHD_16_9_W = 3840 # 8*480
UHD_16_9_H = 2160 # 8*270
UHD_9_16_W = 2160 # 8*270
UHD_9_16_H = 3840 # 8*480

# it is important that the resolution buckets properly cover the training dataset,
# or else that we exclude from the dataset videos that are out of this range
# right now, finetrainers will crash if that happens, so the workaround is to have more buckets in here

NB_FRAMES_1   =          1  # 1
NB_FRAMES_9   = 8      + 1  # 8 + 1
NB_FRAMES_17  = 8 *  2 + 1  # 16 + 1
NB_FRAMES_33  = 8 *  4 + 1  # 32 + 1
NB_FRAMES_49  = 8 *  6 + 1  # 48 + 1
NB_FRAMES_65  = 8 *  8 + 1  # 64 + 1
NB_FRAMES_73  = 8 *  9 + 1  # 72 + 1
NB_FRAMES_81  = 8 * 10 + 1  # 80 + 1
NB_FRAMES_89  = 8 * 11 + 1  # 88 + 1
NB_FRAMES_97  = 8 * 12 + 1  # 96 + 1
NB_FRAMES_105 = 8 * 13 + 1  # 104 + 1
NB_FRAMES_113 = 8 * 14 + 1  # 112 + 1
NB_FRAMES_121 = 8 * 14 + 1  # 120 + 1
NB_FRAMES_129 = 8 * 16 + 1  # 128 + 1
NB_FRAMES_137 = 8 * 16 + 1  # 136 + 1
NB_FRAMES_145 = 8 * 18 + 1  # 144 + 1
NB_FRAMES_161 = 8 * 20 + 1  # 160 + 1
NB_FRAMES_177 = 8 * 22 + 1  # 176 + 1
NB_FRAMES_193 = 8 * 24 + 1  # 192 + 1
NB_FRAMES_201 = 8 * 25 + 1  # 200 + 1
NB_FRAMES_209 = 8 * 26 + 1  # 208 + 1
NB_FRAMES_217 = 8 * 27 + 1  # 216 + 1
NB_FRAMES_225 = 8 * 28 + 1  # 224 + 1
NB_FRAMES_233 = 8 * 29 + 1  # 232 + 1
NB_FRAMES_241 = 8 * 30 + 1  # 240 + 1
NB_FRAMES_249 = 8 * 31 + 1  # 248 + 1
NB_FRAMES_257 = 8 * 32 + 1  # 256 + 1
NB_FRAMES_265 = 8 * 33 + 1  # 264 + 1
NB_FRAMES_273 = 8 * 34 + 1  # 272 + 1
NB_FRAMES_289 = 8 * 36 + 1  # 288 + 1
NB_FRAMES_305 = 8 * 38 + 1  # 304 + 1
NB_FRAMES_321 = 8 * 40 + 1  # 320 + 1
NB_FRAMES_337 = 8 * 42 + 1  # 336 + 1
NB_FRAMES_353 = 8 * 44 + 1  # 352 + 1
NB_FRAMES_369 = 8 * 46 + 1  # 368 + 1
NB_FRAMES_385 = 8 * 48 + 1  # 384 + 1
NB_FRAMES_401 = 8 * 50 + 1  # 400 + 1

# ------ HOW BUCKETS WORK:----------
# Basically, to train or fine-tune a video model with Finetrainers, we need to specify all the possible accepted videos lengths AND size combinations (buckets), in the form: (BUCKET_CONFIGURATION_1, BUCKET_CONFIGURATION_2, ..., BUCKET_CONFIGURATION_N)
# Where a bucket is: (NUMBER_OF_FRAMES_PLUS_ONE, HEIGHT_IN_PIXELS, WIDTH_IN_PIXELS)
# For instance, for 2 seconds of a 1024x576 video at 24 frames per second, plus one frame (I think there is always an extra frame for the initial starting image), we would get:
#   NUMBER_OF_FRAMES_PLUS_ONE = (2*24) + 1 = 48 + 1 = 49
#   HEIGHT_IN_PIXELS = 576
#   WIDTH_IN_PIXELS = 1024
# -> This would give a bucket like this: (49, 576, 1024)
#

SD_TRAINING_BUCKETS = [
    (NB_FRAMES_1,   SD_16_9_H, SD_16_9_W), # 1
    (NB_FRAMES_9,   SD_16_9_H, SD_16_9_W), # 8 + 1
    (NB_FRAMES_17,  SD_16_9_H, SD_16_9_W), # 16 + 1
    (NB_FRAMES_33,  SD_16_9_H, SD_16_9_W), # 32 + 1
    (NB_FRAMES_49,  SD_16_9_H, SD_16_9_W), # 48 + 1
    (NB_FRAMES_65,  SD_16_9_H, SD_16_9_W), # 64 + 1
    (NB_FRAMES_73,  SD_16_9_H, SD_16_9_W), # 72 + 1
    (NB_FRAMES_81,  SD_16_9_H, SD_16_9_W), # 80 + 1
    (NB_FRAMES_89,  SD_16_9_H, SD_16_9_W), # 88 + 1
    (NB_FRAMES_97,  SD_16_9_H, SD_16_9_W), # 96 + 1
    (NB_FRAMES_105, SD_16_9_H, SD_16_9_W), # 104 + 1
    (NB_FRAMES_113, SD_16_9_H, SD_16_9_W), # 112 + 1
    (NB_FRAMES_121, SD_16_9_H, SD_16_9_W), # 121 + 1
    (NB_FRAMES_129, SD_16_9_H, SD_16_9_W), # 128 + 1
    (NB_FRAMES_137, SD_16_9_H, SD_16_9_W), # 136 + 1
    (NB_FRAMES_145, SD_16_9_H, SD_16_9_W), # 144 + 1
    (NB_FRAMES_161, SD_16_9_H, SD_16_9_W), # 160 + 1
    (NB_FRAMES_177, SD_16_9_H, SD_16_9_W), # 176 + 1
    (NB_FRAMES_193, SD_16_9_H, SD_16_9_W), # 192 + 1
    (NB_FRAMES_201, SD_16_9_H, SD_16_9_W), # 200 + 1
    (NB_FRAMES_209, SD_16_9_H, SD_16_9_W), # 208 + 1
    (NB_FRAMES_217, SD_16_9_H, SD_16_9_W), # 216 + 1
    (NB_FRAMES_225, SD_16_9_H, SD_16_9_W), # 224 + 1
    (NB_FRAMES_233, SD_16_9_H, SD_16_9_W), # 232 + 1
    (NB_FRAMES_241, SD_16_9_H, SD_16_9_W), # 240 + 1
    (NB_FRAMES_249, SD_16_9_H, SD_16_9_W), # 248 + 1
    (NB_FRAMES_257, SD_16_9_H, SD_16_9_W), # 256 + 1
    (NB_FRAMES_265, SD_16_9_H, SD_16_9_W), # 264 + 1
    (NB_FRAMES_273, SD_16_9_H, SD_16_9_W), # 272 + 1
]

# For 1280x720 images and videos (from 1 frame up to 272)
MD_TRAINING_BUCKETS = [
    (NB_FRAMES_1,   MD_16_9_H, MD_16_9_W), # 1
    (NB_FRAMES_9,   MD_16_9_H, MD_16_9_W), # 8 + 1
    (NB_FRAMES_17,  MD_16_9_H, MD_16_9_W), # 16 + 1
    (NB_FRAMES_33,  MD_16_9_H, MD_16_9_W), # 32 + 1
    (NB_FRAMES_49,  MD_16_9_H, MD_16_9_W), # 48 + 1
    (NB_FRAMES_65,  MD_16_9_H, MD_16_9_W), # 64 + 1
    (NB_FRAMES_73,  MD_16_9_H, MD_16_9_W), # 72 + 1
    (NB_FRAMES_81,  MD_16_9_H, MD_16_9_W), # 80 + 1
    (NB_FRAMES_89,  MD_16_9_H, MD_16_9_W), # 88 + 1
    (NB_FRAMES_97,  MD_16_9_H, MD_16_9_W), # 96 + 1
    (NB_FRAMES_105, MD_16_9_H, MD_16_9_W), # 104 + 1
    (NB_FRAMES_113, MD_16_9_H, MD_16_9_W), # 112 + 1
    (NB_FRAMES_121, MD_16_9_H, MD_16_9_W), # 121 + 1
    (NB_FRAMES_129, MD_16_9_H, MD_16_9_W), # 128 + 1
    (NB_FRAMES_137, MD_16_9_H, MD_16_9_W), # 136 + 1
    (NB_FRAMES_145, MD_16_9_H, MD_16_9_W), # 144 + 1
    (NB_FRAMES_161, MD_16_9_H, MD_16_9_W), # 160 + 1
    (NB_FRAMES_177, MD_16_9_H, MD_16_9_W), # 176 + 1
    (NB_FRAMES_193, MD_16_9_H, MD_16_9_W), # 192 + 1
    (NB_FRAMES_201, MD_16_9_H, MD_16_9_W), # 200 + 1
    (NB_FRAMES_209, MD_16_9_H, MD_16_9_W), # 208 + 1
    (NB_FRAMES_217, MD_16_9_H, MD_16_9_W), # 216 + 1
    (NB_FRAMES_225, MD_16_9_H, MD_16_9_W), # 224 + 1
    (NB_FRAMES_233, MD_16_9_H, MD_16_9_W), # 232 + 1
    (NB_FRAMES_241, MD_16_9_H, MD_16_9_W), # 240 + 1
    (NB_FRAMES_249, MD_16_9_H, MD_16_9_W), # 248 + 1
    (NB_FRAMES_257, MD_16_9_H, MD_16_9_W), # 256 + 1
    (NB_FRAMES_265, MD_16_9_H, MD_16_9_W), # 264 + 1
    (NB_FRAMES_273, MD_16_9_H, MD_16_9_W), # 272 + 1
]


# Model specific default parameters
# These are used instead of the previous TRAINING_PRESETS

# Resolution buckets for different models
RESOLUTION_OPTIONS = {
    "SD (1024x576)": "SD_TRAINING_BUCKETS",
    "HD (1280x720)": "MD_TRAINING_BUCKETS"
}

# Default parameters for Hunyuan Video
HUNYUAN_VIDEO_DEFAULTS = {
    "lora": {
        "learning_rate": 2e-5,
        "flow_weighting_scheme": "none",
        "lora_rank": DEFAULT_LORA_RANK_STR,
        "lora_alpha": DEFAULT_LORA_ALPHA_STR
    },
    "control-lora": {
        "learning_rate": 2e-5,
        "flow_weighting_scheme": "none",
        "lora_rank": "128",
        "lora_alpha": "128",
        "control_type": "custom",
        "train_qk_norm": True,
        "frame_conditioning_type": "index",
        "frame_conditioning_index": 0,
        "frame_conditioning_concatenate_mask": True
    }
}

# Default parameters for LTX Video
LTX_VIDEO_DEFAULTS = {
    "lora": {
        "learning_rate": DEFAULT_LEARNING_RATE,
        "flow_weighting_scheme": "none",
        "lora_rank": DEFAULT_LORA_RANK_STR,
        "lora_alpha": DEFAULT_LORA_ALPHA_STR
    },
    "full-finetune": {
        "learning_rate": DEFAULT_LEARNING_RATE,
        "flow_weighting_scheme": "logit_normal"
    },
    "control-lora": {
        "learning_rate": DEFAULT_LEARNING_RATE,
        "flow_weighting_scheme": "logit_normal",
        "lora_rank": "128",
        "lora_alpha": "128",
        "control_type": "custom",
        "train_qk_norm": True,
        "frame_conditioning_type": "index",
        "frame_conditioning_index": 0,
        "frame_conditioning_concatenate_mask": True
    }
}

# Default parameters for Wan
WAN_DEFAULTS = {
    "lora": {
        "learning_rate": 5e-5,
        "flow_weighting_scheme": "logit_normal",
        "lora_rank": "32",
        "lora_alpha": "32"
    },
    "control-lora": {
        "learning_rate": 5e-5,
        "flow_weighting_scheme": "logit_normal",
        "lora_rank": "32",
        "lora_alpha": "32",
        "control_type": "custom",
        "train_qk_norm": True,
        "frame_conditioning_type": "index",
        "frame_conditioning_index": 0,
        "frame_conditioning_concatenate_mask": True
    }
}

@dataclass
class TrainingConfig:
    """Configuration class for finetrainers training"""
    
    # Required arguments must come first
    model_name: str
    pretrained_model_name_or_path: str
    data_root: str
    output_dir: str
    
    # Optional arguments follow
    revision: Optional[str] = None
    version: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Dataset arguments

    # note: video_column and caption_column serve a dual purpose,
    # when using the CSV mode they have to be CSV column names,
    # otherwise they have to be filename (relative to the data_root dir path)
    video_column: str = "videos.txt"
    caption_column: str = "prompts.txt"

    id_token: Optional[str] = None
    video_resolution_buckets: List[Tuple[int, int, int]] = field(default_factory=lambda: SD_TRAINING_BUCKETS)
    video_reshape_mode: str = "center"
    caption_dropout_p: float = DEFAULT_CAPTION_DROPOUT_P
    caption_dropout_technique: str = "empty"
    precompute_conditions: bool = False
    
    # Diffusion arguments
    flow_resolution_shifting: bool = False
    flow_weighting_scheme: str = "none"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_mode_scale: float = 1.29
    
    # Training arguments
    training_type: str = "lora"
    seed: int = DEFAULT_SEED
    mixed_precision: str = "bf16"
    batch_size: int = 1
    train_steps: int = DEFAULT_NB_TRAINING_STEPS
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    checkpointing_steps: int = DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS
    checkpointing_limit: Optional[int] = 2
    resume_from_checkpoint: Optional[str] = None
    enable_slicing: bool = True
    enable_tiling: bool = True

    # Optimizer arguments
    optimizer: str = "adamw"
    lr: float = DEFAULT_LEARNING_RATE
    scale_lr: bool = False
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = DEFAULT_NB_LR_WARMUP_STEPS
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-4
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Miscellaneous arguments
    tracker_name: str = "finetrainers"
    report_to: str = "wandb"
    nccl_timeout: int = 1800

    @classmethod
    def hunyuan_video_lora(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for Hunyuan video-to-video LoRA training"""
        return cls(
            model_name="hunyuan_video",
            pretrained_model_name_or_path="hunyuanvideo-community/HunyuanVideo",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=2e-5,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=1,
            lora_rank=DEFAULT_LORA_RANK,
            lora_alpha=DEFAULT_LORA_ALPHA,
            video_resolution_buckets=buckets or SD_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="none",  # Hunyuan specific
            training_type="lora"
        )
    
    @classmethod
    def ltx_video_lora(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for LTX-Video LoRA training"""
        return cls(
            model_name="ltx_video",
            pretrained_model_name_or_path="Lightricks/LTX-Video",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=DEFAULT_LEARNING_RATE,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=4,
            lora_rank=DEFAULT_LORA_RANK,
            lora_alpha=DEFAULT_LORA_ALPHA,
            video_resolution_buckets=buckets or SD_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="logit_normal",  # LTX specific
            training_type="lora"
        )
        
    @classmethod
    def ltx_video_full_finetune(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for LTX-Video full finetune training"""
        return cls(
            model_name="ltx_video",
            pretrained_model_name_or_path="Lightricks/LTX-Video",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=1e-5,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=1,
            video_resolution_buckets=buckets or SD_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="logit_normal",  # LTX specific
            training_type="full-finetune"
        )
        
    @classmethod
    def wan_lora(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for Wan T2V LoRA training"""
        return cls(
            model_name="wan",
            pretrained_model_name_or_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=5e-5,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=1,
            lora_rank=32,
            lora_alpha=32,
            target_modules=["blocks.*(to_q|to_k|to_v|to_out.0)"],  # Wan-specific target modules
            video_resolution_buckets=buckets or SD_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="logit_normal",  # Wan specific
            training_type="lora"
        )

    def to_args_list(self) -> List[str]:
        """Convert config to command line arguments list"""
        args = []
        
        # Model arguments 

        # Add model_name (required argument)
        args.extend(["--model_name", self.model_name])
        
        args.extend(["--pretrained_model_name_or_path", self.pretrained_model_name_or_path])
        if self.revision:
            args.extend(["--revision", self.revision])
        if self.version:
            args.extend(["--variant", self.version]) 
        if self.cache_dir:
            args.extend(["--cache_dir", self.cache_dir])

        # Dataset arguments
        args.extend(["--dataset_config", self.data_root])
        
        # Add ID token if specified
        if self.id_token:
            args.extend(["--id_token", self.id_token])
            
        # Add video resolution buckets
        if self.video_resolution_buckets:
            bucket_strs = [f"{f}x{h}x{w}" for f, h, w in self.video_resolution_buckets]
            args.extend(["--video_resolution_buckets"] + bucket_strs)
            
        args.extend(["--caption_dropout_p", str(self.caption_dropout_p)])
        args.extend(["--caption_dropout_technique", self.caption_dropout_technique])
        if self.precompute_conditions:
            args.append("--precompute_conditions")

        if hasattr(self, 'precomputation_items') and self.precomputation_items:
            args.extend(["--precomputation_items", str(self.precomputation_items)])
            
        # Diffusion arguments
        if self.flow_resolution_shifting:
            args.append("--flow_resolution_shifting")
        args.extend(["--flow_weighting_scheme", self.flow_weighting_scheme])
        args.extend(["--flow_logit_mean", str(self.flow_logit_mean)])
        args.extend(["--flow_logit_std", str(self.flow_logit_std)])
        args.extend(["--flow_mode_scale", str(self.flow_mode_scale)])

        # Training arguments
        args.extend(["--training_type",self.training_type])
        args.extend(["--seed", str(self.seed)])
        
        # We don't use this, because mixed precision is handled by accelerate launch, not by the training script itself.
        #args.extend(["--mixed_precision", self.mixed_precision])
        
        args.extend(["--batch_size", str(self.batch_size)])
        args.extend(["--train_steps", str(self.train_steps)])
        
        # LoRA specific arguments
        if self.training_type == "lora":
            args.extend(["--rank", str(self.lora_rank)])
            args.extend(["--lora_alpha", str(self.lora_alpha)])
            args.extend(["--target_modules"] + self.target_modules)
            
        args.extend(["--gradient_accumulation_steps", str(self.gradient_accumulation_steps)])
        if self.gradient_checkpointing:
            args.append("--gradient_checkpointing")
        args.extend(["--checkpointing_steps", str(self.checkpointing_steps)])
        if self.checkpointing_limit:
            args.extend(["--checkpointing_limit", str(self.checkpointing_limit)])
        if self.resume_from_checkpoint:
            args.extend(["--resume_from_checkpoint", self.resume_from_checkpoint])
        if self.enable_slicing:
            args.append("--enable_slicing")
        if self.enable_tiling:
            args.append("--enable_tiling")

        # Optimizer arguments
        args.extend(["--optimizer", self.optimizer])
        args.extend(["--lr", str(self.lr)])
        if self.scale_lr:
            args.append("--scale_lr")
        args.extend(["--lr_scheduler", self.lr_scheduler])
        args.extend(["--lr_warmup_steps", str(self.lr_warmup_steps)])
        args.extend(["--lr_num_cycles", str(self.lr_num_cycles)])
        args.extend(["--lr_power", str(self.lr_power)])
        args.extend(["--beta1", str(self.beta1)])
        args.extend(["--beta2", str(self.beta2)])
        args.extend(["--weight_decay", str(self.weight_decay)])
        args.extend(["--epsilon", str(self.epsilon)])
        args.extend(["--max_grad_norm", str(self.max_grad_norm)])

        # Miscellaneous arguments
        args.extend(["--tracker_name", self.tracker_name])
        args.extend(["--output_dir", self.output_dir])
        args.extend(["--report_to", self.report_to])
        args.extend(["--nccl_timeout", str(self.nccl_timeout)])

        # normally this is disabled by default, but there was a bug in finetrainers
        # so I had to fix it in trainer.py to make sure we check for push_to-hub
        #args.append("--push_to_hub")
        #args.extend(["--hub_token", str(False)])
        #args.extend(["--hub_model_id", str(False)])

        # If you are using LLM-captioned videos, it is common to see many unwanted starting phrases like
        # "In this video, ...", "This video features ...", etc.
        # To remove a simple subset of these phrases, you can specify
        # --remove_common_llm_caption_prefixes when starting training.
        args.append("--remove_common_llm_caption_prefixes")

        return args
