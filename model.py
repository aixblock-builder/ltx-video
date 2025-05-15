# #model_marketplace.config
# {"framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name":"black-forest-labs/FLUX.1-dev",
#       "value": "black-forest-labs/FLUX.1-dev",
#       "size": 120,
#       "paramasters": "12B",
#       "tflops": 14,
#       "vram": 19, # 16 + 15%
#       "nodes": 1
#     },
#     {
#       "name":"black-forest-labs/FLUX.1-schnell",
#       "value": "black-forest-labs/FLUX.1-schnell",
#       "size": 120,
#       "paramasters": "12B",
#       "tflops": 14,
#       "vram": 19,
#       "nodes": 1
#     },
#     {
#       "name":"multimodalart/FLUX.1-dev2pro-full",
#       "value": "multimodalart/FLUX.1-dev2pro-full",
#       "size": 200,
#       "paramasters": "12B",
#       "tflops": 14,
#       "vram": 40,
#       "nodes": 2
#     },
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
                        pipe = FluxPipeline.from_pretrained(
                            model_id,
                            transformer=model_nf4,
                            torch_dtype=torch.bfloat16,
                            device_map="balanced",
                        )
                        image = pipe(
                            prompt=prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                        ).images[0]
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

        @dataclass
        class Config:
            guidance_scale = 3.0
            step = 5
            width = 512
            height = 512
            prompt = ""

        css = """
            .importantButton {
                background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
                border: none !important;
            }
            .importantButton:hover {
                background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
                border: none !important;
            }
            .disclaimer {font-variant-caps: all-small-caps; font-size: xx-small;}
            .xsmall {font-size: x-small;}
        """

        example_list = [
            "A cinematic split-frame composition where the lens is half-submerged underwater. Below the surface, a serene and calm lakebed stretches out, softly illuminated by refracted light. A nude woman stands gracefully on the sandy lakebed, her pose tranquil, with her long hair drifting weightlessly in the still water. Above the surface, the storm rages ferociously: lightning streaks across the dark clouds, torrential rain pours down, and towering waves crash violently. In the distance, a weathered, ancient lighthouse struggles against the storm, its flickering light barely piercing the chaos. The waterline is sharp, with bubbles and light refractions enhancing the split effect. Dynamic lighting highlights the emotional contrast between the calm underwater world and the savage storm above. (masterpiece, best quality, ultra-detailed, photorealistic, cinematic:1.3) --ar 16:9 --intense-storm --lens-split --weight 2. 0",
            "A collage of various abstract shapes and lines, with red accents, on white paper, composition includes black ink and bold strokes, creating an urban feel, geometric elements in the background, adding to its modern aesthetic, digital illustration technique, contemporary look and feel --ar 1:2 --stylize 750",
            "A close-up shot of a Chernobyl liquidator's gas mask, filling the frame with gritty, realistic detail. The mask is worn and authentic, modeled after Soviet-era designs with rounded lenses, thick rubber seals, and heavy straps, covered in ash and grime from the reactor’s fallout. The lenses are the focal point, each glass surface slightly warped and scratched, reflecting the fierce glow of distant fires within the reactor. Flames dance across the curved lenses in shades of red, orange, and intense yellow, creating a haunting, distorted view of the fiery chaos within. Lighting and Shadow Play: The overall lighting is low and moody, with harsh shadows defining the rugged texture of the mask and highlighting its worn, weathered surface. Dim light from a flickering source to the left illuminates the mask partially, casting deep shadows across the rubber surface, creating an ominous, high-contrast look. Hazy backlighting subtly outlines the mask’s contours, adding depth and a sense of foreboding. Atmospheric Details: The air is thick with smoke and radioactive dust, faintly illuminated by the fiery reflection in the lenses. Tiny, glowing particles float through the air, adding to the toxic, dangerous atmosphere. Thin wisps of smoke drift around the mask, softening the edges and giving the scene a ghostly quality. Surface Texture and Wear: The rubber of the mask is cracked and stained, showing the toll of exposure to radiation and extreme heat. Ash and small flecks of debris cling to its surface, adding realism and a gritty feel. Around the edges, faint condensation gathers on the rubber, hinting at the liquidator’s breath inside the suit. Reflection Details in the Lenses: In the mask's lenses, we see reflections of distant fires raging inside the reactor, with structures burning and twisted metal faintly visible in the intense glow. The reflections are slightly distorted, warped by the rounded glass, as if the fires themselves are bending reality. Occasional flickers of light pulse in the reflection, conveying the flickering intensity of the flames. Mood and Composition: The close-up shot emphasizes the isolation, courage, and silent determination of the liquidator. The composition is hauntingly intimate, placing the viewer face-to-face with the mask, capturing the intensity of the task and the immense, invisible danger surrounding them. Every detail contributes to a heavy, foreboding atmosphere, evoking a sense of dread and silent resilience.",
            "A vast, cosmic Yggdrasil, the World Tree, stretches into the star-speckled void of space, its massive trunk and sprawling branches connecting the nine realms in Norse mythology. The tree itself appears ancient, with weathered bark and roots that dig deep into the surrounding universe, each level alive with its own distinct world: At the highest branches, a golden realm glows with an ethereal light. Majestic halls with shining towers rise among the branches, surrounded by a shimmering bridge that arcs through the stars, connecting this upper realm to the central trunk. Nearby, a lush, untamed world flourishes with dense forests and flowing rivers, brimming with natural beauty and tranquility. Otherworldly beings, graceful and radiant, can be seen moving through this paradise, embodying peace and balance. A level lower, radiant beings with light and beauty in their form dwell in an enchanted forest, their world illuminated by soft, otherworldly light. Their realm is vibrant, with trees that sparkle as if dusted with stardust, and rivers that flow with a gentle, magical glow. The air is filled with a soft luminescence, creating an almost dreamlike atmosphere. At the central trunk, a rugged, mountainous land stretches across the roots and branches, where humans reside in familiar landscapes of rolling hills, seas, and forests. Humans gaze up at the towering Yggdrasil in awe, their settlements and villages nestled in its shadow, looking like fragile worlds of their own amidst the tree’s vast presence. Nearby, on the tree's rugged branches, hulking beings with immense strength roam amidst icy, craggy mountains and dark forests. Their world is filled with jagged peaks, frozen rivers, and heavy snow, giving the sense of a fierce and untamed wilderness. Towering fortresses built into the cliffs loom ominously, while giant figures peer down from their snow-covered perches, casting an air of silent menace. Further down, twisted roots encircle a dark, cavernous world filled with deep shadows and molten light from subterranean forges. Small but sturdy beings, master craftsmen, toil here, creating intricate weapons and treasures. The dim light reflects off pools of molten metal, illuminating their forge workshops and complex machinery in the gloom. The air is thick with smoke and the glow of embers, giving this level a feeling of constant labor and creation. Below, in a realm shrouded in mist and ice, an icy wasteland extends endlessly, cloaked in a veil of thick fog. Massive glaciers and frozen rivers twist through the foggy landscape, with serpentine creatures moving within the ice, only half-seen beneath the surface. The air is frigid, and ghostly figures appear through the mist, creating an atmosphere of deathly stillness and isolation. Opposite this icy realm, a world ablaze with fire and molten lava seethes with destructive energy. Colossal beings made of fire guard this realm, wielding massive, flame-covered weapons. Rivers of molten fire wind through scorched ground, erupting sporadically as if the very earth is alive with fury. The entire realm glows with a fierce, red-orange light, casting shadows of immense heat and danger. At the deepest root of Yggdrasil, a desolate, shadowed world stretches endlessly. Souls drift aimlessly through a barren landscape, their ghostly forms barely visible through a dark, hazy fog. A distant, imposing structure looms over this shadowed land, giving it a sense of quiet despair and finality, as if all life here has been stripped of hope. Around Yggdrasil: At the very roots of the tree, a massive serpent coils and gnaws, its dark scales glistening, a symbol of decay gnawing away at life. At the topmost branches, a fierce eagle with piercing eyes keeps watch over the realms, embodying wisdom and vigilance. Darting up and down the tree’s trunk is a mischievous squirrel, carrying messages (or perhaps insults) between the eagle above and the serpent below, adding a touch of lively movement and drama to the scene. Atmosphere and Depth: The entire tree glows with a faint, cosmic light, as if alive and breathing. Particles of stardust drift around it, highlighting the different realms in hues of gold, green, blue, and red. In the foreground, fragments of ancient runes hover, casting a dim glow over the scene, anchoring each realm’s unique appearance in an aura of mythic wonder. The cosmos stretches infinitely in the background, with stars and distant galaxies giving depth and scale, making Yggdrasil appear both infinite and ancient.",
            "astronaut woman, barefoot with perfect foot perfect toe, full body view. Blue short hair, belly piercing, tank top, space suit orange pants, science fiction style rifle. Alien stile starship setting. Searching a xenomorph in the dark corridors with the rifle armed. Graceful and muscular pose, canvas-like style with hyperrealistic details, high resolution for intricate elements, digital illustration with vivid colors, dramatic lighting to enhance the scene.",
            "A bold and vibrant modern illustration of a beautiful woman leaning towards her reflection in a sleek, frameless mirror. The surface of the mirror ripples like water where her lips meet it for a kiss, creating concentric waves that distort her reflection in a mesmerizing, surreal effect. The reflection appears alive and dynamic, almost as if reaching back to her, enhancing the emotional depth and intrigue of the scene. The woman’s hair flows softly, with a few loose strands catching the light, adding movement to the composition. Her features are illuminated with striking, modern lighting, creating a radiant glow that highlights the contours of her face and shoulders. The background is abstract and minimal, with soft gradients of electric blue and golden hues blending seamlessly, adding a surreal, dreamlike quality. The rippling effect of the mirror is intricate, with delicate reflections of light breaking across the watery surface. Subtle glowing particles and soft, diffused highlights surround the scene, enhancing the magical realism.",
            "A dark, gritty comic-style illustration, rich with hand-drawn textures, heavy inking, and a worn, weathered aesthetic. On the jagged, desolate surface of the moon, three astronauts in scuffed, retrofuturistic red spacesuits sprint for their lives, kicking up clouds of lunar dust that trail behind them. Their sleek, Soviet-inspired spacesuits are dull and battered, with faded USSR insignias barely visible under scratches and grime. Each astronaut is armed, firing crude, makeshift weapons backward in desperation as they attempt to fend off their alien attackers. In the distance, an ominous alien spacecraft hovers above the lunar horizon, its massive, angular silhouette casting long shadows across the surface. Bright neon-green plasma bolts streak through the darkness, fired from the ship’s glowing, turret-like weapons. The plasma bolts illuminate the gritty scene in brief, blinding flashes, casting jagged shadows and reflecting off the astronauts' scratched visors. The composition is chaotic and dynamic, with the lead astronaut crouched and firing while the others sprint, their postures tense and frantic. One astronaut stumbles, his weapon raised as he looks back in horror at the attackers. The moon's surface is jagged and uneven, littered with sharp rocks, deep craters, and faint traces of long-forgotten alien ruins etched with strange, glowing glyphs. The alien ship is vast and angular, with faint lights along its hull giving it a menacing presence. The Earth looms faintly in the background, partially obscured by lunar dust and darkness. The atmosphere is tense and moody, dominated by muted greys, dusty reds, and bright flashes of neon green from the plasma fire. The illustration is gritty and imperfect, with visible hand-drawn lines, bold inking, and heavy shadows. The texture of the lunar dust and the weathered suits is palpable, creating a tactile, raw aesthetic. The scene feels alive with motion and desperation, capturing the chaotic action of a life-or-death struggle in a hostile, alien world",
            "An exquisite 8K Ultra HD double exposure image, featuring a majestic lion silhouette seamlessly blended with a vivid African forest sunrise. The lion's details are intricately incorporated into the landscape, creating a stunning visual effect. The monochrome background highlights the lion's white fur, while the sharp focus and crisp lines showcase the incredible level of detail. The full color of the lion contrasts with the white background, evoking a sense of awe and wonder. The overall effect is cinematic, capturing the essence of a breathtaking African sunset. , illustration, photo, cinematic, typography, 3d render",
            "[Abstract style waterfalls, wildlife] inside the silhouette of a [woman] âs head that is a double exposure photograph . Non-representational, colors and shapes, expression of feelings, imaginative, highly detailed",
            "A shimmering, translucent wall of liquid-like energy rises from the ground, stretching endlessly into the sky. It hums softly, its surface rippling with iridescent waves of blue, violet, and silver, casting faint reflections onto the terrain around it. The veil divides two worlds: on one side, a vibrant jungle teeming with life. Towering trees with lush, emerald canopies sway gently, their leaves glowing faintly. Exotic creatures with iridescent scales and translucent wings dart between the branches, their colors flashing like living jewels. Streams of crystalline water cascade down ancient rocks, pooling in pristine, reflective ponds, while luminous plants pulse softly in rhythmic harmony. On the other side lies a barren wasteland under a blood-red sky. Cracked earth stretches into the distance, scarred with jagged canyons and dotted with skeletal remnants of a once-thriving world. Blackened, twisted spires rise from the ground, and an oppressive heat radiates from the ground, distorting the air. Lightning forks across the sky, illuminating the scorched terrain for fleeting moments. At the edge of the veil stands a lone figure, their silhouette illuminated by the glowing energy. Their hand hovers just above the surface, fingers outstretched as if daring to touch it. The two realities—one vibrant and alive, the other desolate and broken—are mirrored in their wide, mesmerized eyes. The figure’s stance is tense, caught in a moment of wonder and indecision, their presence the only bridge between the two worlds. The air around the veil crackles faintly, shimmering with barely contained energy. Small tendrils of light curl outward from its surface, brushing against the figure and the ground like ethereal whispers. Fine particles of dust and pollen drift lazily in the light of the jungle, contrasting with the barren emptiness of the wasteland. The scene is vivid and layered, a profound juxtaposition of creation and destruction, framed by the ethereal glow of the veil",
        ]

        STATS_DEFAULT = SimpleNamespace(llm=None, config=Config())

        # TODO: add trained checkpoint list into model_list
        # def get_checkpoint_list(project):
        #     pass

        def generate_btn_handler(
            prompt: str, guidance_scale: float, step: int, width: int, height: int
        ) -> tuple:
            if prompt == "" or prompt is None:
                raise Exception("Prompt cannot be empty")
            if pipe_demo is None:
                raise Exception(
                    "Please load the model first by clicking 'Load Model' button"
                )

            image = pipe_demo(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=step,
                guidance_scale=guidance_scale,
            ).images[0]

            return image, ""

        with gr.Blocks(
            theme=gr.themes.Soft(text_size="sm"),
            title="Flux Image Generator",
            css=css,
        ) as demo_txt_to_img:
            stats = gr.State(STATS_DEFAULT)

            with gr.Row():
                with gr.Column(scale=3):
                    image_field = gr.Image(label="Output Image", elem_id="output_image")
                with gr.Column(scale=1):
                    load_model_btn = gr.Button("Load Model", variant="primary")
                    status_box = gr.Textbox(
                        label="Model Status",
                        interactive=False,
                        value="Model not loaded",
                        lines=2,
                    )
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.TextArea(
                        label="Prompt:",
                        value=example_list[0],
                        elem_id="small-textarea",
                        lines=10,
                        max_lines=8,
                    )
                    generate_btn = gr.Button("Generate", interactive=False)
                with gr.Column(scale=1):
                    guidance_scale = gr.Slider(
                        value=STATS_DEFAULT.config.guidance_scale,
                        minimum=0.0,
                        maximum=30.0,
                        step=0.1,
                        label="Guidance scale",
                    )
                    step = gr.Slider(
                        value=STATS_DEFAULT.config.step,
                        minimum=3,
                        maximum=100,
                        step=10,
                        label="Step",
                    )
                    width = gr.Number(
                        value=STATS_DEFAULT.config.width,
                        label="Image width (64-1920)",
                        precision=0,
                        minimum=64,
                        maximum=1920,
                        interactive=True,
                    )
                    height = gr.Number(
                        value=STATS_DEFAULT.config.width,
                        label="Image height (64-1080)",
                        precision=0,
                        minimum=64,
                        maximum=1080,
                        interactive=True,
                    )

            with gr.Accordion("Example inputs", open=True):
                examples = gr.Examples(
                    examples=example_list,
                    inputs=[prompt],
                    examples_per_page=60,
                )

            def load_model_handler():
                yield "Loading model, please wait...", gr.update(interactive=False)
                status, btn_update = load_model_fn()
                if "successfully" in status:
                    yield status, gr.update(interactive=True)
                else:
                    yield status, gr.update(interactive=False)

            # Event handlers
            generate_btn.click(
                fn=generate_btn_handler,
                inputs=[prompt, guidance_scale, step, width, height],
                outputs=[image_field, prompt],
                api_name="generate",
            )

            load_model_btn.click(
                fn=load_model_handler,
                inputs=[],
                outputs=[status_box, generate_btn],
                api_name=None,
                queue=True,  # Bắt buộc để enable yield (stream trạng thái)
            )

        with gr.Blocks(css=css) as demo:
            gr.Markdown("Flux Image Generator")
            with gr.Tabs():
                # if task == "text-to-image":
                with gr.Tab(label=task):
                    demo_txt_to_img.render()

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
