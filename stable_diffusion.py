#!/usr/bin/python3

import os
import time
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

default_model_id = "prompthero/openjourney-v4"
device = "cpu"  # Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: rocm


def get_inputs(prompt, batch_size=1):
    generator = [torch.Generator(device).manual_seed(i)
                 for i in range(batch_size)]
    prompts = batch_size * [prompt]
    

    return {"prompt": prompts, "generator": generator}


def callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")

    with torch.no_grad():
        #pipe.upcast_vae()
        latents = latents.to(
            next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        images = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type='pil')

        plt.figure()
        plt.imshow(images[0])
        plt.show()

    return callback_kwargs


def main():
    model_id = input("model [prompthero/openjourney-v4]: ")
    if not model_id:
        model_id = default_model_id
    print(f"Loading Model {model_id}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config)

    pipeline = pipeline.to(device)

    prompt = input("prompt: ")

    out_path = f"./{int(time.time())}"
    os.makedirs(out_path)
    i = 0
    print(f"Generating Images")
    images = pipeline(callback_on_step_end=callback,
                      num_inference_steps = 20,
                      **get_inputs(prompt=prompt, batch_size=4)
                      ).images
    for image in images:
        image.save(f"{out_path}/{i}.png")
        i += 1

    print(f"Saved output images to {out_path}")


if __name__ == "__main__":
    main()
