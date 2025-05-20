import gc
import random
from datetime import datetime

import gradio as gr
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image


# ------------------------------------------------------------------------------
def round_to_nearest_resolution_acceptable_by_vae(
    height, width, pipe: LTXConditionPipeline
):
    height = height - (height % pipe.vae_temporal_compression_ratio)
    width = width - (width % pipe.vae_temporal_compression_ratio)
    return height, width


def generate(
    model,
    model_upsample,
    img2vid_image=None,
    prompt="",
    negative_prompt="",
    frame_rate=25,
    num_inference_steps=30,
    denoise_strength=0.4,
    height=512,
    width=768,
    num_frames=121,
    progress=gr.Progress(),
):

    pipe = LTXConditionPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        model_upsample,
        vae=pipe.vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe_upsample.to("cuda")
    pipe.vae.enable_tiling()

    conditions_list = None
    if img2vid_image:
        image = load_image(img2vid_image)
        video = [image]
        condition1 = LTXVideoCondition(video=video, frame_index=0)
        conditions_list = [condition1]

    downscale_factor = 2 / 3

    # Part 1. Generate video at smaller resolution
    downscaled_height, downscaled_width = int(height * downscale_factor), int(
        width * downscale_factor
    )
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width
    )
    latents = pipe(
        conditions=conditions_list,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(0),
        output_type="latent",
    ).frames

    # Part 2. Upscale generated video using latent upsampler with fewer inference steps
    # The available latent upsampler upscales the height/width by 2x
    upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames

    # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
    video = pipe(
        conditions=conditions_list,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=upscaled_width,
        height=upscaled_height,
        num_frames=num_frames,
        denoise_strength=denoise_strength,  # Effectively, 4 inference steps out of 10
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(0),
        output_type="pil",
    ).frames[0]

    # Part 4. Downscale the video to the expected resolution
    video = [frame.resize((width, height)) for frame in video]

    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = random.randint(1000, 9999)
    output_filename = f"output_{timestamp}_{random_suffix}.mp4"

    export_to_video(video, output_filename, fps=frame_rate)
    del pipe, pipe_upsample
    torch.cuda.empty_cache()
    gc.collect()
    return output_filename


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


def create_advanced_options():
    with gr.Accordion("Advanced Options (Optional)", open=False):
        inference_steps = gr.Slider(
            label="Inference Steps", minimum=1, maximum=50, step=1, value=30
        )
        denoise_strength = gr.Slider(
            label="Denoise Strength", minimum=1.0, maximum=5.0, step=0.1, value=0.4
        )

        height_slider = gr.Slider(
            label="Height",
            minimum=256,
            maximum=1024,
            step=64,
            value=512,
            visible=True,
        )
        width_slider = gr.Slider(
            label="Width",
            minimum=256,
            maximum=1024,
            step=64,
            value=768,
            visible=True,
        )
        num_frames_slider = gr.Slider(
            label="Number of Frames",
            minimum=1,
            maximum=200,
            step=1,
            value=97,
            visible=True,
        )

        return [
            inference_steps,
            denoise_strength,
            height_slider,
            width_slider,
            num_frames_slider,
        ]


# def preset_changed(preset):
# if preset != "Custom":
#     selected = next(item for item in preset_options if item["label"] == preset)
#     return (
#         selected["height"],
#         selected["width"],
#         selected["num_frames"],
#         gr.update(visible=True),
#         gr.update(visible=True),
#         gr.update(visible=True),
#     )
# else:
#     return (
#         None,
#         None,
#         None,
#         gr.update(visible=True),
#         gr.update(visible=True),
#         gr.update(visible=True),
#     )


css = """
#col-container {
    margin: 0 auto;
    max-width: 1220px;
}
"""


def create_demo(
    model="Lightricks/LTX-Video-0.9.7-dev",
    model_upsample="Lightricks/ltxv-spatial-upscaler-0.9.7",
):
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

                txt2vid_negative_prompt = gr.Textbox(
                    label="Enter Negative Prompt",
                    placeholder="Describe what you don't want in the video...",
                    value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                    lines=2,
                )

            with gr.Column():
                # Di chuyển 2 ô hiển thị model và model_upsample đến đây
                model_textbox = gr.Textbox(
                    label="Model", value=model, interactive=False
                )
                model_upsample_textbox = gr.Textbox(
                    label="Upsample Model", value=model_upsample, interactive=False
                )

                # txt2vid_preset = gr.Dropdown(
                #     choices=[p["label"] for p in preset_options] + ["Custom"],
                #     value="Custom",
                #     label="Choose Resolution Preset",
                # )

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

                txt2vid_output = gr.Video(label="Generated Output")

        with gr.Row():
            gr.Examples(
                examples=[
                    [
                        "A young woman in a traditional Mongolian dress is peeking through a sheer white curtain, her face showing a mix of curiosity and apprehension. The woman has long black hair styled in two braids, adorned with white beads, and her eyes are wide with a hint of surprise. Her dress is a vibrant blue with intricate gold embroidery, and she wears a matching headband with a similar design. The background is a simple white curtain, which creates a sense of mystery and intrigue.ith long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage",
                        "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                    ],
                    [
                        "A young man with blond hair wearing a yellow jacket stands in a forest and looks around. He has light skin and his hair is styled with a middle part. He looks to the left and then to the right, his gaze lingering in each direction. The camera angle is low, looking up at the man, and remains stationary throughout the video. The background is slightly out of focus, with green trees and the sun shining brightly behind the man. The lighting is natural and warm, with the sun creating a lens flare that moves across the man's face. The scene is captured in real-life footage.",
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

        txt2vid_generate.click(
            fn=generate,
            inputs=[
                model_textbox,
                model_upsample_textbox,
                img2vid_image,
                txt2vid_prompt,
                txt2vid_negative_prompt,
                txt2vid_frame_rate,
                *txt2vid_advanced,
            ],
            outputs=txt2vid_output,
            concurrency_limit=1,
            concurrency_id="generate_video",
            queue=True,
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)
