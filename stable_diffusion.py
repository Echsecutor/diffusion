#!/usr/bin/python3

import os
import time
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

default_model_id = "prompthero/openjourney-v4"
default_guidance_scale = 12
default_num_inference_steps = 20
device = "cpu"  # Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: rocm

out_path = "./"
seeds = []


def callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")

    with torch.no_grad():
        latents = latents.to(
            next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        images = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type='pil')

        fig, axs = plt.subplots(len(images), 1)
        fig.suptitle(f'Images generated at step_index = {step_index}, timestep = {timestep}')

        for i in range(len(images)):
            axs[i].imshow(images[i])
            axs[i].set_title(f'Seed {seeds[i]}')
        
        plt.show()

        plt.savefig(f'{out_path}/tmp_{{seeds[i]}}_{timestep}.png')

    return callback_kwargs


def main():
    global out_path, seeds
    model_id = input(f"model [{default_model_id}]: ")
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
    negative_prompt = input("negative_prompt: ")
    num_inference_steps = input(
        f"num_inference_steps [{default_num_inference_steps}]: ")
    if not num_inference_steps:
        num_inference_steps = default_num_inference_steps
    num_inference_steps = int(num_inference_steps)

    seed = "-1"

    print("Enter the seeds for the images to be generated.")
    print("The number of seeds equals the number of generated images.")
    print("Press enter when no more seeds should be added.")
    while seed != "":
        seed = input("Seed: ")
        if seed != "":
            seeds.append(int(seed))

    guidance_scale = input(
        f"guidance_scale (1-20) [{default_guidance_scale}]: ")
    if not guidance_scale:
        guidance_scale = default_guidance_scale
    guidance_scale = float(guidance_scale)

    out_path = f"./{int(time.time())}"
    os.makedirs(out_path)

    print(f"Generating Images")
    images = pipeline(callback_on_step_end=callback,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale,
                      generator=[torch.Generator(
                          device).manual_seed(i) for i in seeds],
                      negative_prompt=len(seeds) * [negative_prompt],
                      prompt=len(seeds) * [prompt]
                      ).images
    i = 0
    for image in images:
        image.save(f"{out_path}/{seeds[i]}.png")
        i += 1

    print(f"Saved output images to {out_path}")


if __name__ == "__main__":
    main()
