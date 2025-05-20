# #model_marketplace.config
# {"framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name":"Lightricks LTX-Video",
#       "value": "Lightricks/LTX-Video",
#       "size": 200,
#       "paramasters": "12B",
#       "tflops": 30,
#       "vram": 42, # 16 + 15%
#       "nodes": 1
#     }
#   ], "cuda": "11.4", "task":["text-to-image"]}

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Dict, List, Optional, get_type_hints

import gradio as gr
from gradio_toggle import Toggle
import torch
import wandb
import yaml
from aixblock_ml.model import AIxBlockMLBase
from centrifuge import (
    CentrifugeError,
    Client,
    ClientEventHandler,
    SubscriptionEventHandler,
)
from datasets import load_dataset
from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from huggingface_hub import HfApi, HfFolder, hf_hub_download, login
from mcp.server.fastmcp import FastMCP

import constants as const
import utils
from dashboard import promethus_grafana
from function_ml import (
    connect_project,
    download_dataset,
    upload_checkpoint_mixed_folder,
)
from logging_class import start_queue, write_log
from misc import get_device_count
from param_class import TrainingConfigFlux, TrainingConfigFluxLora
from loguru import logger
import gc
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

import imageio
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod


MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257



def load_image_to_tensor_with_resize_and_crop(
    image_path, target_height=512, target_width=768
):
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


# Generate output video name
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
# --------------------------------------------------------------------------------------------
with open("models.yaml", "r") as file:
    models = yaml.safe_load(file)

mcp = FastMCP("aixblock-mcp")


def base64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def generate_jwt(user, channel=""):
    """Note, in tests we generate token on client-side - this is INSECURE
    and should not be used in production. Tokens must be generated on server-side."""
    hmac_secret = "d0a70289-9806-41f6-be6d-f4de5fe298fb"  # noqa: S105 - this is just a secret used in tests.
    header = {"typ": "JWT", "alg": "HS256"}
    payload = {"sub": user}
    if channel:
        # Subscription token
        payload["channel"] = channel
    encoded_header = base64url_encode(json.dumps(header).encode("utf-8"))
    encoded_payload = base64url_encode(json.dumps(payload).encode("utf-8"))
    signature_base = encoded_header + b"." + encoded_payload
    signature = hmac.new(
        hmac_secret.encode("utf-8"), signature_base, hashlib.sha256
    ).digest()
    encoded_signature = base64url_encode(signature)
    jwt_token = encoded_header + b"." + encoded_payload + b"." + encoded_signature
    return jwt_token.decode("utf-8")


async def get_client_token() -> str:
    return generate_jwt("42")


async def get_subscription_token(channel: str) -> str:
    return generate_jwt("42", channel)


class ClientEventLoggerHandler(ClientEventHandler):
    async def on_connected(self, ctx):
        logging.info("Connected to server")


class SubscriptionEventLoggerHandler(SubscriptionEventHandler):
    async def on_subscribed(self, ctx):
        logging.info("Subscribed to channel")


def setup_client(channel_log):
    client = Client(
        "wss://rt.aixblock.io/centrifugo/connection/websocket",
        events=ClientEventLoggerHandler(),
        get_token=get_client_token,
        use_protobuf=False,
    )

    sub = client.new_subscription(
        channel_log,
        events=SubscriptionEventLoggerHandler(),
        # get_token=get_subscription_token,
    )

    return client, sub


async def send_log(sub, log_message):
    try:
        await sub.publish(data={"log": log_message})
    except CentrifugeError as e:
        logging.error("Error publish: %s", e)


async def send_message(sub, message):
    try:
        await sub.publish(data=message)
    except CentrifugeError as e:
        logging.error("Error publish: %s", e)


async def log_training_progress(sub, log_message):
    await send_log(sub, log_message)


def run_train(command, channel_log="training_logs"):
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client, sub = setup_client(channel_log)

            async def main():
                await client.connect()
                await sub.subscribe()
                await log_training_progress(sub, "Training started")
                log_file_path = "logs/llm-ddp.log"
                last_position = 0  # Vị trí đã đọc đến trong file log
                await log_training_progress(sub, "Training training")
                promethus_grafana.promethus_push_to("training")

                while True:
                    try:
                        current_size = os.path.getsize(log_file_path)
                        if current_size > last_position:
                            with open(log_file_path, "r") as log_file:
                                log_file.seek(last_position)
                                new_lines = log_file.readlines()
                                # print(new_lines)
                                for line in new_lines:
                                    print("--------------", f"{line.strip()}")
                                    #             # Thay thế đoạn này bằng code để gửi log
                                    await log_training_progress(sub, f"{line.strip()}")
                            last_position = current_size

                        time.sleep(5)
                    except Exception as e:
                        print(e)

                # promethus_grafana.promethus_push_to("finish")
                # await log_training_progress(sub, "Training completed")
                # await client.disconnect()
                # loop.stop()

            try:
                loop.run_until_complete(main())
            finally:
                loop.close()  # Đảm bảo vòng lặp được đóng lại hoàn toàn

        except Exception as e:
            print(e)

    thread_start = threading.Thread(target=run)
    thread_start.start()
    subprocess.run(command, shell=True)
    # try:
    #     promethus_grafana.promethus_push_to("finish")
    # except:
    #     pass


def fetch_logs(channel_log="training_logs"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client, sub = setup_client(channel_log)

    async def run():
        await client.connect()
        await sub.subscribe()
        history = await sub.history(limit=-1)
        logs = []
        for pub in history.publications:
            log_message = pub.data.get("log")
            if log_message:
                logs.append(log_message)
        await client.disconnect()
        return logs

    return loop.run_until_complete(run())


# deprecated, for sd-scripts
def download(base_model, train_config):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # download unet
    if "pretrained_model_name_or_path" not in train_config:
        if "FLUX.1-dev" in base_model or "FLUX.1-schnell" in base_model:
            unet_folder = const.MODELS_DIR.joinpath("unet")
        else:
            unet_folder = const.MODELS_DIR.joinpath("unet/{repo}")
        unet_path = unet_folder.joinpath(model_file)
        if not unet_path.exists():
            unet_folder.mkdir(parents=True, exist_ok=True)
            print(f"download {base_model}")
            hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)
        train_config["pretrained_model_name_or_path"] = str(unet_path)

    # download vae
    if "ae" not in train_config:
        vae_folder = const.MODELS_DIR.joinpath("vae")
        vae_path = vae_folder.joinpath("ae.sft")
        if not vae_path.exists():
            vae_folder.mkdir(parents=True, exist_ok=True)
            print(f"downloading ae.sft...")
            hf_hub_download(
                repo_id="cocktailpeanut/xulf-dev",
                local_dir=vae_folder,
                filename="ae.sft",
            )
        train_config["ae"] = str(vae_path)

    # download clip
    if "clip_l" not in train_config:
        clip_folder = const.MODELS_DIR.joinpath("clip")
        clip_l_path = clip_folder.joinpath("clip_l.safetensors")
        if not clip_l_path.exists():
            clip_folder.mkdir(parents=True, exist_ok=True)
            print(f"download clip_l.safetensors")
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                local_dir=clip_folder,
                filename="clip_l.safetensors",
            )
        train_config["clip_l"] = str(clip_l_path)

    # download t5xxl
    if "t5xxl" not in train_config:
        t5xxl_path = clip_folder.joinpath("t5xxl_fp16.safetensors")
        if not t5xxl_path.exists():
            print(f"download t5xxl_fp16.safetensors")
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                local_dir=clip_folder,
                filename="t5xxl_fp16.safetensors",
            )
        train_config["t5xxl"] = str(t5xxl_path)

    return train_config


class MyModel(AIxBlockMLBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        HfFolder.save_token(const.HF_TOKEN)
        login(token=const.HF_ACCESS_TOKEN)
        wandb.login("allow", const.WANDB_TOKEN)
        print("Login successful")

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
            print("CUDA is available.")
        else:
            print("No GPU available, using CPU.")

        try:
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_properties(0).major
                if compute_capability > 8:
                    self.torch_dtype = torch.bfloat16
                elif compute_capability > 7:
                    self.torch_dtype = torch.float16
            else:
                self.torch_dtype = None  # auto setup for < 7
        except Exception as e:
            self.torch_dtype = None

        try:
            n_gpus = torch.cuda.device_count()
            _ = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
        except Exception as e:
            print("Cannot get cuda memory:", e)
            _ = 0
        max_memory = {i: _ for i in range(n_gpus)}
        print("max memory:", max_memory)

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """ """
        print(
            f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}"""
        )
        return []

    def fit(self, event, data, **kwargs):
        """ """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")

    @mcp.tool()
    def action(self, command, **kwargs):
        """
        {
            "command": "train",
            "params": {
                "project_id": 432,
                "framework": "huggingface",
                "model_id": "black-forest-labs/FLUX.1-dev",
                "push_to_hub": true,
                "push_to_hub_token": "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",
                // "dataset_id": 13,
                "TrainingArguments": {
                    // see param_class.py for the full training arguments
                    ...
                }
            },
            "project": "1"
        }
        """
        # region Train
        if command.lower() == "train":
            try:
                clone_dir = const.CLONE_DIR
                framework = kwargs.get("framework", const.FRAMEWORK)
                task = kwargs.get("task", const.TASK)
                world_size = kwargs.get("world_size", const.WORLD_SIZE)
                rank = kwargs.get("rank", const.RANK)
                master_add = kwargs.get("master_add", const.MASTER_ADDR)
                master_port = kwargs.get("master_port", const.MASTER_PORT)

                host_name = kwargs.get("host_name", const.HOST_NAME)
                token = kwargs.get("token", const.AXB_TOKEN)
                wandb_api_key = kwargs.get("wantdb_api_key", const.WANDB_TOKEN)

                training_arguments = kwargs.get("TrainingArguments", {})
                project_id = kwargs.get("project_id", None)
                model_id = kwargs.get("model_id", const.MODEL_ID)
                dataset_id = kwargs.get("dataset_id", None)
                push_to_hub = kwargs.get("push_to_hub", const.PUSH_TO_HUB)
                push_to_hub_token = kwargs.get("push_to_hub_token", const.HF_TOKEN)
                channel_log = kwargs.get("channel_log", const.CHANNEL_LOGS)

                training_arguments.setdefault("lora", False)
                training_arguments.setdefault("pretrained_model_name_or_path", model_id)
                training_arguments.setdefault("resolution", const.RESOLUTION)
                training_arguments.setdefault("instance_prompt", const.PROMPT)

                log_queue, _ = start_queue(channel_log)
                write_log(log_queue)

                HfFolder.save_token(push_to_hub_token)
                login(token=push_to_hub_token)
                if len(wandb_api_key) > 0 and wandb_api_key != const.WANDB_TOKEN:
                    wandb.login("allow", wandb_api_key)

                os.environ["TORCH_USE_CUDA_DSA"] = "1"

                def func_train_model():
                    project = connect_project(host_name, token, project_id)
                    print("Connect project:", project)

                    zip_dir = os.path.join(clone_dir, "data_zip")
                    extract_dir = os.path.join(clone_dir, "extract")
                    os.makedirs(zip_dir, exist_ok=True)
                    os.makedirs(extract_dir, exist_ok=True)

                    dataset_name = training_arguments.get("dataset_name", const.DATASET)
                    if dataset_name and dataset_id is None:
                        training_arguments["dataset_name"] = dataset_name

                    # only process dataset from s3. hf dataset is processed inside train_dreambooth_... .py
                    # works only for instance_prompt, prior-preservation loss method should be done differently
                    if dataset_id and isinstance(dataset_id, int):
                        project = connect_project(host_name, token, project_id)
                        dataset_name = download_dataset(project, dataset_id, zip_dir)
                        print(dataset_name)
                        if dataset_name:
                            data_zip_dir = os.path.join(zip_dir, dataset_name)
                            with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                utils.clean_folder(extract_dir)
                                zip_ref.extractall(path=extract_dir)

                        # special handle for exported s3 json file
                        json_file, json_file_dir = utils.get_first_json_file(
                            extract_dir
                        )
                        if json_file and utils.is_platform_json_file(
                            json_file, json_file_dir.parent
                        ):
                            with open(json_file_dir) as f:
                                jsonl_1 = json.load(f)
                                jsonl_2 = [
                                    {
                                        "image": data["data"].get("images"),
                                        "prompt": data.get("prompt"),
                                    }
                                    for data in jsonl_1
                                ]
                                with open(json_file_dir, "w") as f:
                                    json.dump(jsonl_2, f)
                                print("modified json to usable format")

                        dataset_name = dataset_name.replace(".zip", "")
                        try:
                            ds = load_dataset("imagefolder", data_dir=extract_dir)
                        except Exception as e:
                            ds = load_dataset(extract_dir)

                        dataset_dir = const.DATASETS_DIR.joinpath(str(dataset_name))
                        dataset_dir.mkdir(parents=True, exist_ok=True)
                        folder_list = utils.create_local_dataset(
                            ds, dataset_dir, training_arguments
                        )
                        training_arguments["instance_data_dir"] = str(
                            dataset_dir.joinpath("images")
                        )

                    output_dir = const.OUTPUTS_DIR.joinpath(dataset_name)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    training_arguments["output_dir"] = str(output_dir)

                    if framework == "huggingface":
                        print("torch.cuda.device_count()", torch.cuda.device_count())
                        if world_size > 1:
                            if int(rank) == 0:
                                print("master node")
                            else:
                                print("worker node")

                        if torch.cuda.device_count() > 1:  # multi gpu
                            compute_mode = "--multi_gpu"
                            n_process = world_size * torch.cuda.device_count()

                        elif torch.cuda.device_count() == 1:  # 1 gpu
                            compute_mode = ""
                            n_process = world_size * torch.cuda.device_count()

                        else:  # no gpu
                            compute_mode = "--cpu"
                            n_process = torch.get_num_threads()

                        if training_arguments["lora"] is False:
                            filtered_configs = utils.filter_config_arguments(
                                training_arguments, TrainingConfigFlux
                            )
                        else:
                            filtered_configs = utils.filter_config_arguments(
                                training_arguments, TrainingConfigFluxLora
                            )

                        json_file = const.PROJ_DIR.joinpath(const.JSON_TRAINING_ARGS)
                        with open(json_file, "w") as f:
                            json.dump(asdict(filtered_configs), f)

                        #  --dynamo_backend 'no' \
                        # --rdzv_backend c10d
                        command = (
                            "accelerate launch {compute_mode} \
                                --main_process_ip {head_node_ip} \
                                --main_process_port {port} \
                                --num_machines {SLURM_NNODES} \
                                --num_processes {num_processes}\
                                --machine_rank {rank} \
                                --num_cpu_threads_per_process 1 \
                                {file_name} \
                                --training_args_json {json_file} \
                                {push_to_hub} \
                                --hub_token {push_to_hub_token} \
                                --channel_log {channel_log} "
                        ).format(
                            file_name=(
                                "./train_dreambooth_flux.py"
                                if not training_arguments["lora"]
                                else "./train_dreambooth_lora_flux.py"
                            ),
                            compute_mode=compute_mode,
                            head_node_ip=master_add,
                            port=master_port,
                            SLURM_NNODES=world_size,
                            num_processes=n_process,
                            rank=rank,
                            json_file=str(json_file),
                            push_to_hub="--push_to_hub" if push_to_hub else "",
                            push_to_hub_token=push_to_hub_token,
                            channel_log=channel_log,
                        )

                        command = " ".join(command.split())
                        print(command)
                        subprocess.run(
                            command,
                            shell=True,
                            # capture_output=True, text=True).stdout.strip("\n")
                        )

                    else:
                        raise Exception("Unimplemented framework behavior:", framework)

                    print(push_to_hub)
                    if push_to_hub:
                        import datetime

                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y%m%d")
                        time_str = now.strftime("%H%M%S")
                        version = f"{date_str}-{time_str}"
                        upload_checkpoint_mixed_folder(project, version, output_dir)

                import threading

                train_thread = threading.Thread(target=func_train_model)
                train_thread.start()
                return {"message": "train started successfully"}

            except Exception as e:
                return {"message": f"train failed: {e}"}

        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "./train_dreambooth_flux.py"])
            return {"message": "train stop successfully", "result": "Done"}

        elif command.lower() == "tensorboard":

            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(
                    f"tensorboard --logdir /app/data/logs --host 0.0.0.0 --port=6006",
                    stdout=subprocess.PIPE,
                    stderr=None,
                    shell=True,
                )
                out = p.communicate()
                print(out)

            import threading

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        # region Predict
        elif command.lower() == "predict":
            try:
                prompt = kwargs.get("prompt", None)
                model_id = kwargs.get("model_id", "black-forest-labs/FLUX.1-dev")
                chkpt_name = kwargs.get("checkpoint", None)
                width = kwargs.get("width", 1024)
                height = kwargs.get("height", 1024)
                num_inference_steps = kwargs.get("num_inference_steps", 4)
                guidance_scale = kwargs.get("guidance_scale", 2)
                format = kwargs.get("format", "JPEG")

                predictions = []

                if prompt == "" or prompt is None:
                    return None, ""
                
                with torch.no_grad():
                    try:
                        import tempfile
                        img2vid_image="",
                        prompt="",
                        txt2vid_analytics_toggle=False,
                        negative_prompt="",
                        frame_rate=25,
                        seed=0,
                        num_inference_steps=30,
                        guidance_scale=3,
                        height=512,
                        width=768,
                        num_frames=121,
                        args = {
                            "ckpt_dir": "Lightricks/LTX-Video",
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "frame_rate": frame_rate,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "seed": 0,
                            "output_path": os.path.join(tempfile.gettempdir(), "gradio"),
                            "num_images_per_prompt": 1,
                            "input_image_path": img2vid_image,
                            "input_video_path": "",
                            "bfloat16": True,
                            "disable_load_needed_only": False
                        }
                        logger.warning(f"Running generation with arguments: {args}")
                        image = self.predict_action(args=args)
                    except Exception as e:
                        logger.error(str(e))

                buffered = BytesIO()
                image.save(buffered, format=format)
                image.save(const.PROJ_DIR.joinpath(f"image.{format}"))
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                generated_url = f"/downloads?path=image.{format}"

                del pipe, model_nf4
                gc.collect()
                torch.cuda.empty_cache()

                result = {
                    "model_version": model_id,
                    "result": {
                        "format": format,
                        "image": img_base64,
                        "image_url": generated_url,
                    },
                }

                logger.info(result)

                predictions.append({
                    'result': [{
                        'from_name': "generated_text",
                        'to_name': "text_output",
                        'type': 'textarea',
                        'value': {
                            'text': [img_base64],
                            "generated_url": generated_url
                        }
                    }],
                    'model_version': ""
                })

                return {"message": "predict completed successfully", "result": predictions}

            except Exception as e:
                print(e)
                return {"message": "predict failed", "result": None}

        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "text-to-image":
                prompt_text = f"""
                A planet, yarn art style
                """

            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        else:
            return {"message": "command not supported", "result": None}

            # return {"message": "train completed successfully"}

    def model(self, **kwargs):
        global pipe_demo, model_nf4
        # initialize
        task = kwargs.get("task", "text-to-image")
        model_id = kwargs.get("model_id", "black-forest-labs/FLUX.1-dev")
        # Initialize pipe as None - will be loaded when button is clicked
        pipe_demo = None
        model_nf4 = None
        hf_access_token = kwargs.get("hf_access_token", const.HF_ACCESS_TOKEN)
        login(token=hf_access_token)
        from huggingface_hub import snapshot_download
        model_path = './data/checkpoint'   # The local directory to save downloaded checkpoint
       
       
        def load_model_fn():
            global pipe_demo, model_nf4
            if model_nf4 is not None:
                # If the model has already been loaded, do not load it again
                return "Model already loaded!", gr.update(interactive=True)
            try:
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
                return "Model loaded successfully!", gr.update(interactive=True)
            except Exception as e:
                return f"Error loading model: {str(e)}", gr.update(interactive=False)
        def preset_changed(preset):
                preset_options = [
                {"label": "1216x704, 41 frames", "width": 1216, "height": 704, "num_frames": 41},
                {"label": "1088x704, 49 frames", "width": 1088, "height": 704, "num_frames": 49},
                {"label": "1056x640, 57 frames", "width": 1056, "height": 640, "num_frames": 57},
                {"label": "992x608, 65 frames", "width": 992, "height": 608, "num_frames": 65},
                {"label": "896x608, 73 frames", "width": 896, "height": 608, "num_frames": 73},
                {"label": "896x544, 81 frames", "width": 896, "height": 544, "num_frames": 81},
                {"label": "832x544, 89 frames", "width": 832, "height": 544, "num_frames": 89},
                {"label": "800x512, 97 frames", "width": 800, "height": 512, "num_frames": 97},
                {"label": "768x512, 97 frames", "width": 768, "height": 512, "num_frames": 97},
                {"label": "800x480, 105 frames", "width": 800, "height": 480, "num_frames": 105},
                {"label": "736x480, 113 frames", "width": 736, "height": 480, "num_frames": 113},
                {"label": "704x480, 121 frames", "width": 704, "height": 480, "num_frames": 121},
                {"label": "704x448, 129 frames", "width": 704, "height": 448, "num_frames": 129},
                {"label": "672x448, 137 frames", "width": 672, "height": 448, "num_frames": 137},
                {"label": "640x416, 153 frames", "width": 640, "height": 416, "num_frames": 153},
                {"label": "672x384, 161 frames", "width": 672, "height": 384, "num_frames": 161},
                {"label": "640x384, 169 frames", "width": 640, "height": 384, "num_frames": 169},
                {"label": "608x384, 177 frames", "width": 608, "height": 384, "num_frames": 177},
                {"label": "576x384, 185 frames", "width": 576, "height": 384, "num_frames": 185},
                {"label": "608x352, 193 frames", "width": 608, "height": 352, "num_frames": 193},
                {"label": "576x352, 201 frames", "width": 576, "height": 352, "num_frames": 201},
                {"label": "544x352, 209 frames", "width": 544, "height": 352, "num_frames": 209},
                {"label": "512x352, 225 frames", "width": 512, "height": 352, "num_frames": 225},
                {"label": "512x352, 233 frames", "width": 512, "height": 352, "num_frames": 233},
                {"label": "544x320, 241 frames", "width": 544, "height": 320, "num_frames": 241},
                {"label": "512x320, 249 frames", "width": 512, "height": 320, "num_frames": 249},
                {"label": "512x320, 257 frames", "width": 512, "height": 320, "num_frames": 257},
            ] 
                if preset != "Custom":
                    selected = next(item for item in preset_options if item["label"] == preset)
                    return (
                        selected["height"],
                        selected["width"],
                        selected["num_frames"],
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )
                else:
                    return (
                        None,
                        None,
                        None,
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                    )
        def create_advanced_options():
                with gr.Accordion("Advanced Options (Optional)", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=1000000, step=1, value=0)
                    inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=30)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, step=0.1, value=3.0)

                    height_slider = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=1024,
                        step=64,
                        value=512,
                        visible=False,
                    )
                    width_slider = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=1024,
                        step=64,
                        value=768,
                        visible=False,
                    )
                    num_frames_slider = gr.Slider(
                        label="Number of Frames",
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=97,
                        visible=False,
                    )

                    return [
                        seed,
                        inference_steps,
                        guidance_scale,
                        height_slider,
                        width_slider,
                        num_frames_slider,
                    ]
        preset_options = [
                {"label": "1216x704, 41 frames", "width": 1216, "height": 704, "num_frames": 41},
                {"label": "1088x704, 49 frames", "width": 1088, "height": 704, "num_frames": 49},
                {"label": "1056x640, 57 frames", "width": 1056, "height": 640, "num_frames": 57},
                {"label": "992x608, 65 frames", "width": 992, "height": 608, "num_frames": 65},
                {"label": "896x608, 73 frames", "width": 896, "height": 608, "num_frames": 73},
                {"label": "896x544, 81 frames", "width": 896, "height": 544, "num_frames": 81},
                {"label": "832x544, 89 frames", "width": 832, "height": 544, "num_frames": 89},
                {"label": "800x512, 97 frames", "width": 800, "height": 512, "num_frames": 97},
                {"label": "768x512, 97 frames", "width": 768, "height": 512, "num_frames": 97},
                {"label": "800x480, 105 frames", "width": 800, "height": 480, "num_frames": 105},
                {"label": "736x480, 113 frames", "width": 736, "height": 480, "num_frames": 113},
                {"label": "704x480, 121 frames", "width": 704, "height": 480, "num_frames": 121},
                {"label": "704x448, 129 frames", "width": 704, "height": 448, "num_frames": 129},
                {"label": "672x448, 137 frames", "width": 672, "height": 448, "num_frames": 137},
                {"label": "640x416, 153 frames", "width": 640, "height": 416, "num_frames": 153},
                {"label": "672x384, 161 frames", "width": 672, "height": 384, "num_frames": 161},
                {"label": "640x384, 169 frames", "width": 640, "height": 384, "num_frames": 169},
                {"label": "608x384, 177 frames", "width": 608, "height": 384, "num_frames": 177},
                {"label": "576x384, 185 frames", "width": 576, "height": 384, "num_frames": 185},
                {"label": "608x352, 193 frames", "width": 608, "height": 352, "num_frames": 193},
                {"label": "576x352, 201 frames", "width": 576, "height": 352, "num_frames": 201},
                {"label": "544x352, 209 frames", "width": 544, "height": 352, "num_frames": 209},
                {"label": "512x352, 225 frames", "width": 512, "height": 352, "num_frames": 225},
                {"label": "512x352, 233 frames", "width": 512, "height": 352, "num_frames": 233},
                {"label": "544x320, 241 frames", "width": 544, "height": 320, "num_frames": 241},
                {"label": "512x320, 249 frames", "width": 512, "height": 320, "num_frames": 249},
                {"label": "512x320, 257 frames", "width": 512, "height": 320, "num_frames": 257},
            ] 
        css="""
        #col-container {
            margin: 0 auto;
            max-width: 1220px;
        }
        """
        
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column():
                    img2vid_image = gr.Image(
                        type="filepath",
                        label="Upload Input Image",
                        elem_id="image_upload",
                    )

                    txt2vid_prompt = gr.Textbox(
                        label="Enter Your Prompt",
                        placeholder="Describe the video you want to generate (minimum 50 characters)...",
                        value="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage.",
                        lines=5,
                    )
                    
                    txt2vid_analytics_toggle = Toggle(
                        label="torchao.quantization.",
                        value=False,
                        interactive=True,
                    )

                    txt2vid_negative_prompt = gr.Textbox(
                        label="Enter Negative Prompt",
                        placeholder="Describe what you don't want in the video...",
                        value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                        lines=2,
                    )

                    txt2vid_preset = gr.Dropdown(
                        choices=[p["label"] for p in preset_options],
                        value="768x512, 97 frames",
                        label="Choose Resolution Preset",
                    )

                    txt2vid_frame_rate = gr.Slider(
                        label="Frame Rate",
                        minimum=21,
                        maximum=30,
                        step=1,
                        value=25,
                    )

                    txt2vid_advanced = create_advanced_options()
                    
                    txt2vid_generate = gr.Button(
                        "Generate Video",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column():
                    txt2vid_output = gr.Video(label="Generated Output")

            with gr.Row():
                gr.Examples(
                    examples=[
                        [
                            "A young woman in a traditional Mongolian dress is peeking through a sheer white curtain, her face showing a mix of curiosity and apprehension. The woman has long black hair styled in two braids, adorned with white beads, and her eyes are wide with a hint of surprise. Her dress is a vibrant blue with intricate gold embroidery, and she wears a matching headband with a similar design. The background is a simple white curtain, which creates a sense of mystery and intrigue.ith long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair’s face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                        ],
                        [
                            "A young man with blond hair wearing a yellow jacket stands in a forest and looks around. He has light skin and his hair is styled with a middle part. He looks to the left and then to the right, his gaze lingering in each direction. The camera angle is low, looking up at the man, and remains stationary throughout the video. The background is slightly out of focus, with green trees and the sun shining brightly behind the man. The lighting is natural and warm, with the sun creating a lens flare that moves across the man’s face. The scene is captured in real-life footage.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                        ],
                        [
                            "A cyclist races along a winding mountain road. Clad in aerodynamic gear, he pedals intensely, sweat glistening on his brow. The camera alternates between close-ups of his determined expression and wide shots of the breathtaking landscape. Pine trees blur past, and the sky is a crisp blue. The scene is invigorating and competitive.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                        ],
                    ],
                    inputs=[txt2vid_prompt, txt2vid_negative_prompt, txt2vid_output],
                    label="Example Text-to-Video Generations",
                )

            txt2vid_preset.change(fn=preset_changed, inputs=[txt2vid_preset], outputs=txt2vid_advanced[3:])
            def ltx_video_generate(
                    img2vid_image="",
                    prompt="",
                    txt2vid_analytics_toggle=False,
                    negative_prompt="",
                    frame_rate=25,
                    seed=0,
                    num_inference_steps=30,
                    guidance_scale=3,
                    height=512,
                    width=768,
                    num_frames=121,
            ):
                import tempfile
                logger = logging.get_logger(__name__)
                logger.info("Starting LTX Video generation...")
                snapshot_download("Lightricks/LTX-Video", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model',token="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")
                snapshot_download(
                    repo_id="stabilityai/stable-diffusion-2-1",
                    repo_type="model",
                    allow_patterns=["unet/*"],  
                    local_dir= "./data/checkpoint/Lightricks/LTX-Video/unet",
                    token="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"
                )
                args = {
                            "ckpt_dir": "Lightricks/LTX-Video",
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "frame_rate": frame_rate,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "seed": 0,
                            "output_path": os.path.join(tempfile.gettempdir(), "gradio"),
                            "num_images_per_prompt": 1,
                            "input_image_path": img2vid_image,
                            "input_video_path": "",
                            "bfloat16": True,
                            "disable_load_needed_only": False
                        }
                logger.warning(f"Running generation with arguments: {args}")

                seed_everething(args['seed'])

                output_dir = (
                    Path(args['output_path'])
                    if args['output_path']
                    else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                # Load image
                if args['input_image_path']:
                    media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                        args['input_image_path'], args['height'], args['width']
                    )
                else:
                    media_items_prepad = None

                height = args['height'] if args['height'] else media_items_prepad.shape[-2]
                width = args['width'] if args['width'] else media_items_prepad.shape[-1]
                num_frames = args['num_frames']

                if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
                    logger.warning(
                        f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, it is suggested to use the resolution below {MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES}."
                    )

                # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
                height_padded = ((height - 1) // 32 + 1) * 32
                width_padded = ((width - 1) // 32 + 1) * 32
                num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

                padding = calculate_padding(height, width, height_padded, width_padded)

                logger.warning(
                    f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
                )

                if media_items_prepad is not None:
                    media_items = F.pad(
                        media_items_prepad, padding, mode="constant", value=-1
                    )  # -1 is the value for padding since the image is normalized to -1, 1
                else:
                    media_items = None

                # Paths for the separate mode directories
                ckpt_dir = "./data/checkpoint/Lightricks/LTX-Video" #Path(args['ckpt_dir'])
                unet_dir =ckpt_dir /  "unet"
                vae_dir = ckpt_dir / "vae"
                scheduler_dir = ckpt_dir / "scheduler"
                

                def load_vae(vae_dir):
                    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
                    vae_config_path = vae_dir / "config.json"
                    with open(vae_config_path, "r") as f:
                        vae_config = json.load(f)
                    vae = CausalVideoAutoencoder.from_config(vae_config)
                    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
                    vae.load_state_dict(vae_state_dict)
                    if torch.cuda.is_available():
                        vae = vae.cuda()
                    return vae.to(torch.bfloat16)


                def load_unet(unet_dir):
                    # hf_hub_download(repo_id="Lightricks/LTX-Video", local_dir="./data/checkpoint/unet", filename="")
                    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
                    unet_config_path = unet_dir / "config.json"
                    transformer_config = Transformer3DModel.load_config(unet_config_path)
                    transformer = Transformer3DModel.from_config(transformer_config)
                    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
                    transformer.load_state_dict(unet_state_dict, strict=True)
                    if torch.cuda.is_available():
                        transformer = transformer.cuda()
                    return transformer


                def load_scheduler(scheduler_dir):
                    scheduler_config_path = scheduler_dir / "scheduler_config.json"
                    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
                    return RectifiedFlowScheduler.from_config(scheduler_config)
                # Load models
                vae = load_vae(vae_dir)
                unet = load_unet(unet_dir)
                scheduler = load_scheduler(scheduler_dir)
                patchifier = SymmetricPatchifier(patch_size=1)
                text_encoder = T5EncoderModel.from_pretrained(
                    "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
                ).to(torch.bfloat16)

                # if torch.cuda.is_available():
                #     text_encoder = text_encoder.to("cuda")

                tokenizer = T5Tokenizer.from_pretrained(
                    "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
                )

                if args['bfloat16'] and unet.dtype != torch.bfloat16:
                    unet = unet.to(torch.bfloat16)

                # Use submodels for the pipeline
                submodel_dict = {
                    "transformer": unet,
                    "patchifier": patchifier,
                    "text_encoder": text_encoder,
                    "tokenizer": tokenizer,
                    "scheduler": scheduler,
                    "vae": vae,
                }

                pipeline = LTXVideoPipeline(**submodel_dict)
                if torch.cuda.is_available() and args['disable_load_needed_only']:
                    pipeline = pipeline.to("cuda")

                # Prepare input for the pipeline
                sample = {
                    "prompt": args['prompt'],
                    "prompt_attention_mask": None,
                    "negative_prompt": args['negative_prompt'],
                    "negative_prompt_attention_mask": None,
                    "media_items": media_items,
                }

                generator = torch.Generator(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ).manual_seed(args['seed'])

                images = pipeline(
                    num_inference_steps=args['num_inference_steps'],
                    num_images_per_prompt=args['num_images_per_prompt'],
                    guidance_scale=args['guidance_scale'],
                    generator=generator,
                    output_type="pt",
                    callback_on_step_end=None,
                    height=height_padded,
                    width=width_padded,
                    num_frames=num_frames_padded,
                    frame_rate=args['frame_rate'],
                    **sample,
                    is_video=True,
                    vae_per_channel_normalize=True,
                    conditioning_method=(
                        ConditioningMethod.FIRST_FRAME
                        if media_items is not None
                        else ConditioningMethod.UNCONDITIONAL
                    ),
                    mixed_precision=not args['bfloat16'],
                    load_needed_only=not args['disable_load_needed_only']
                ).images

                # Crop the padded images to the desired resolution and number of frames
                (pad_left, pad_right, pad_top, pad_bottom) = padding
                pad_bottom = -pad_bottom
                pad_right = -pad_right
                if pad_bottom == 0:
                    pad_bottom = images.shape[3]
                if pad_right == 0:
                    pad_right = images.shape[4]
                images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

                for i in range(images.shape[0]):
                    # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
                    video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
                    # Unnormalizing images to [0, 255] range
                    video_np = (video_np * 255).astype(np.uint8)
                    fps = args['frame_rate']
                    height, width = video_np.shape[1:3]
                    # In case a single image is generated
                    if video_np.shape[0] == 1:
                        output_filename = get_unique_filename(
                            f"image_output_{i}",
                            ".png",
                            prompt=args['prompt'],
                            seed=args['seed'],
                            resolution=(height, width, num_frames),
                            dir=output_dir,
                        )
                        imageio.imwrite(output_filename, video_np[0])
                    else:
                        if args['input_image_path']:
                            base_filename = f"img_to_vid_{i}"
                        else:
                            base_filename = f"text_to_vid_{i}"
                        output_filename = get_unique_filename(
                            base_filename,
                            ".mp4",
                            prompt=args['prompt'],
                            seed=args['seed'],
                            resolution=(height, width, num_frames),
                            dir=output_dir,
                        )

                        # Write video
                        with imageio.get_writer(output_filename, fps=fps) as video:
                            for frame in video_np:
                                video.append_data(frame)

                        # Write condition image
                        if args['input_image_path']:
                            reference_image = (
                                (
                                    media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy()
                                    + 1.0
                                )
                                / 2.0
                                * 255
                            )
                            imageio.imwrite(
                                get_unique_filename(
                                    base_filename,
                                    ".png",
                                    prompt=args['prompt'],
                                    seed=args['seed'],
                                    resolution=(height, width, num_frames),
                                    dir=output_dir,
                                    endswith="_condition",
                                ),
                                reference_image.astype(np.uint8),
                            )
                    logger.warning(f"Output saved {output_filename}")
                    return output_filename

            txt2vid_generate.click(
                fn=ltx_video_generate,
                inputs=[
                    img2vid_image,
                    txt2vid_prompt,
                    txt2vid_analytics_toggle,
                    txt2vid_negative_prompt,
                    txt2vid_frame_rate,
                    *txt2vid_advanced,
                ],
                outputs=txt2vid_output,
                concurrency_limit=1,
                concurrency_id="generate_video",
                queue=True,
            )

            gradio_app, local_url, share_url = demo.launch(
                share=True,
                quiet=True,
                prevent_thread_lock=True,
                server_name="0.0.0.0",
                show_error=True,
            )

        return {"share_url": share_url, "local_url": local_url}

    # deprecated?
    def model_trial(self, project, **kwargs):
        import gradio as gr

        return {"message": "Done", "result": "Done"}

    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)

    def preview(self):
        pass

    def toolbar_predict(self):
        pass

    def toolbar_predict_sam(self):
        pass
    def predict_action(args) -> str:

        logger = logging.get_logger(__name__)

        # args = parser.parse_args()

        logger.warning(f"Running generation with arguments: {args}")

        seed_everething(args.seed)

        output_dir = (
            Path(args.output_path)
            if args.output_path
            else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        if args.input_image_path:
            media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                args.input_image_path, args.height, args.width
            )
        else:
            media_items_prepad = None

        height = args.height if args.height else media_items_prepad.shape[-2]
        width = args.width if args.width else media_items_prepad.shape[-1]
        num_frames = args.num_frames

        if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
            logger.warning(
                f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, it is suggested to use the resolution below {MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES}."
            )

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

        padding = calculate_padding(height, width, height_padded, width_padded)

        logger.warning(
            f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
        )

        if media_items_prepad is not None:
            media_items = F.pad(
                media_items_prepad, padding, mode="constant", value=-1
            )  # -1 is the value for padding since the image is normalized to -1, 1
        else:
            media_items = None

        # Paths for the separate mode directories
        ckpt_dir = Path(args.ckpt_dir)
        unet_dir = ckpt_dir / "unet"
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"
        def load_vae(vae_dir):
            vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
            vae_config_path = vae_dir / "config.json"
            with open(vae_config_path, "r") as f:
                vae_config = json.load(f)
            vae = CausalVideoAutoencoder.from_config(vae_config)
            vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
            vae.load_state_dict(vae_state_dict)
            if torch.cuda.is_available():
                vae = vae.cuda()
            return vae.to(torch.bfloat16)


        def load_unet(unet_dir):
            unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
            unet_config_path = unet_dir / "config.json"
            transformer_config = Transformer3DModel.load_config(unet_config_path)
            transformer = Transformer3DModel.from_config(transformer_config)
            unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
            transformer.load_state_dict(unet_state_dict, strict=True)
            if torch.cuda.is_available():
                transformer = transformer.cuda()
            return transformer


        def load_scheduler(scheduler_dir):
            scheduler_config_path = scheduler_dir / "scheduler_config.json"
            scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
            return RectifiedFlowScheduler.from_config(scheduler_config)
        # Load models
        vae = load_vae(vae_dir)
        unet = load_unet(unet_dir)
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        )
        if torch.cuda.is_available():
            text_encoder = text_encoder.to("cuda")
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )

        if args.bfloat16 and unet.dtype != torch.bfloat16:
            unet = unet.to(torch.bfloat16)

        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": unet,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }

        pipeline = LTXVideoPipeline(**submodel_dict)
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

        # Prepare input for the pipeline
        sample = {
            "prompt": args.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": args.negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": media_items,
        }

        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(args.seed)

        images = pipeline(
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.num_images_per_prompt,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=args.frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=(
                ConditioningMethod.FIRST_FRAME
                if media_items is not None
                else ConditioningMethod.UNCONDITIONAL
            ),
            mixed_precision=not args.bfloat16,
        ).images

        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]
        images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

        for i in range(images.shape[0]):
            # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
            video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
            # Unnormalizing images to [0, 255] range
            video_np = (video_np * 255).astype(np.uint8)
            fps = args.frame_rate
            height, width = video_np.shape[1:3]
            # In case a single image is generated
            if video_np.shape[0] == 1:
                output_filename = get_unique_filename(
                    f"image_output_{i}",
                    ".png",
                    prompt=args.prompt,
                    seed=args.seed,
                    resolution=(height, width, num_frames),
                    dir=output_dir,
                )
                imageio.imwrite(output_filename, video_np[0])
            else:
                if args.input_image_path:
                    base_filename = f"img_to_vid_{i}"
                else:
                    base_filename = f"text_to_vid_{i}"
                output_filename = get_unique_filename(
                    base_filename,
                    ".mp4",
                    prompt=args.prompt,
                    seed=args.seed,
                    resolution=(height, width, num_frames),
                    dir=output_dir,
                )

                # Write video
                with imageio.get_writer(output_filename, fps=fps) as video:
                    for frame in video_np:
                        video.append_data(frame)

                # Write condition image
                if args.input_image_path:
                    reference_image = (
                        (
                            media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy()
                            + 1.0
                        )
                        / 2.0
                        * 255
                    )
                    imageio.imwrite(
                        get_unique_filename(
                            base_filename,
                            ".png",
                            prompt=args.prompt,
                            seed=args.seed,
                            resolution=(height, width, num_frames),
                            dir=output_dir,
                            endswith="_condition",
                        ),
                        reference_image.astype(np.uint8),
                    )
            logger.warning(f"Output saved to {output_dir}")
            return output_filename
    