import base64
import csv
import imghdr
import shutil
from io import BytesIO
from pathlib import Path
from typing import Optional, get_type_hints

import requests
from datasets import load_dataset
from PIL import Image

import constants as const


def filter_config_arguments(config, config_class):
    type_hints = get_type_hints(config_class)
    filtered_args = {
        k: v
        for k, v in config.items()
        if k in type_hints and isinstance(v, type_hints[k])
    }
    return config_class(**filtered_args)


def is_platform_json_file(filename, folder_path):
    # TODO: use different approach other than filename
    files = [f for f in folder_path.iterdir() if f.is_file()]
    print(filename == "result.json", len(files) == 1)
    if filename == "result.json" and len(files) == 1:
        return True
    else:
        return False


def clean_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return
    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()  # Delete file or symbolic link
            elif item.is_dir():
                shutil.rmtree(item)  # Delete folder and its contents
        except Exception as e:
            print(f"Failed to delete {item}: {e}")


def get_first_json_file(folder_path):
    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.is_file() and file.suffix == ".json":
            return file.name, file
    return None, None


def get_image(image_input):
    if isinstance(image_input, Image.Image):
        return image_input

    elif isinstance(image_input, bytes):
        try:
            return Image.open(BytesIO(image_input))
        except Exception:
            raise ValueError("Invalid image bytes")

    elif isinstance(image_input, str):
        # Check if it's a URL
        if image_input.startswith("http"):
            try:
                response = requests.get(image_input, stream=True)
                response.raise_for_status()
                return Image.open(response.raw)
            except requests.RequestException:
                raise ValueError("Invalid image URL")

        # Check if it's Base64 encoded
        elif image_input.startswith("data:image"):
            try:
                header, encoded = image_input.split(",", 1)  # Remove header
                decoded_bytes = base64.b64decode(encoded, validate=True)
                if imghdr.what(None, decoded_bytes) is None:
                    raise ValueError("Invalid Base64 image")
                return Image.open(BytesIO(decoded_bytes))
            except (ValueError, TypeError):
                raise ValueError("Invalid Base64 image data")

        else:
            raise ValueError("Unsupported image format")

    else:
        raise TypeError("Input must be a URL, Base64 string, bytes, or PIL Image")


def create_local_dataset(dataset, dataset_dir, config):
    image_column = config.get("image_column", const.IMG_COL)
    prompt_column = config.get("caption_column", const.PROMPT_COL)
    prompt = config["instance_prompt"]
    resolution = config["resolution"]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = dataset_dir.joinpath("images")
    subset_dir.mkdir(parents=True, exist_ok=True)
    # metadata = []

    for subset in dataset.keys():
        # dataset_dir.mkdir(parents=True, exist_ok=True)
        # subset_dir = dataset_dir.joinpath(subset)
        # subset_dir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(dataset[subset]):
            image = get_image(item[image_column])
            img_name = f"{i}.jpg"
            img_path = subset_dir.joinpath(img_name)
            if isinstance(image, Image.Image):
                image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                image.save(img_path)
            elif isinstance(image, str):
                img_path.write_bytes(Path(image).read_bytes())

            prompt = prompt
            if prompt_column in item:
                prompt = str(item[prompt_column] if item[prompt_column] else prompt)
            # prompt_path = subset_dir.joinpath(f"{i}.txt")
            # with open(prompt_path, "w", encoding="utf-8") as f:
            #     f.write(prompt)
            # metadata.append({"image": img_name, "prompt": prompt})

    # with open(subset_dir.joinpath('metadata.csv'), 'w',) as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['file_name', 'text'])
    #     for item in metadata:
    #         writer.writerow([item['image'], item['prompt']])

    return list("images")


def gen_toml(folder_list, dataset_dir, config):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {config['resolution']}
batch_size = 1
keep_tokens = 1

[[datasets.subsets]]
image_dir = '{(dataset_dir.joinpath('images'))}'
class_tokens = '{config['instance_prompt']}'
num_repeats = 1
"""
    return toml


# def gen_toml(
#   folder_list,
#   dataset_dir,
#   class_tokens,
#   config
# ):
#     toml = f"""[general]
# shuffle_caption = false
# caption_extension = '.txt'
# keep_tokens = 1

# [[datasets]]
# resolution = {config['resolution']}
# batch_size = 1
# keep_tokens = 1

# """
#     for item in folder_list:
#         toml += f"""
#   [[datasets.subsets]]
#   image_dir = '{(dataset_dir.joinpath(item))}'
#   class_tokens = '{class_tokens}'
# num_repeats = 1
# """
#     return toml
