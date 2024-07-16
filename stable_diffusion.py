#!/usr/bin/python3

import os
import time
import matplotlib.pyplot as plt
import json 
from argparse import ArgumentParser


from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

default_model_id = "prompthero/openjourney-v4"
default_guidance_scale = 12
default_num_inference_steps = 20
default_negative_prompt= "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"

device = "cpu"  # Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: rocm

parameters=dict()


def callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")

    with torch.no_grad():
        latents = latents.to(
            next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        images = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type='pil')

        fig, axs = plt.subplots(1, len(images), figsize=(30, 30))
        fig.suptitle(f'Images generated at step_index = {step_index}, timestep = {timestep}')

        for i in range(len(images)):
            axs[i].imshow(images[i])
            axs[i].set_title(f'Seed {parameters["seeds"][i]}')
        
        plt.tight_layout()
        #plt.show()

        plt.savefig(f'{parameters['out_path']}/step_index_{step_index}.png')
        
        print(f"callback_kwargs={callback_kwargs}")

    return callback_kwargs

def parameters_to_pipeline_args(parameters):
    return {'num_inference_steps': parameters['num_inference_steps'],
            'guidance_scale': parameters['guidance_scale'],
            'generator': [torch.Generator(device).manual_seed(i) for i in parameters['seeds']],
            'negative_prompt': len(parameters['seeds']) * [parameters['negative_prompt']],
            'prompt': len(parameters['seeds']) * [parameters['prompt']]
            }

def get_parameters():
    global parameters
    
    parser = ArgumentParser( prog='Exes diffusion CI', description='Run a stable diffusion pipeline to generate images')
    parser.add_argument("-p", "--parameters", help="Read parameters from json FILE instead of querying interactively", metavar="FILE")
    
    args = parser.parse_args()
    
    if args.parameters:
        with open(args.parameters) as parameter_file:
            parameters = json.load(parameter_file)
        return
    
    parameters['model_id'] = input(f"model [{default_model_id}]: ")
    if not parameters['model_id']:
        parameters['model_id'] = default_model_id
    
    parameters['prompt'] = input("prompt: ")
    parameters['negative_prompt'] = input(f"negative_prompt [{default_negative_prompt}]: ")
    if not parameters['negative_prompt']:
        parameters['negative_prompt']=default_negative_prompt
    
    parameters['num_inference_steps'] = input(
        f"num_inference_steps [{default_num_inference_steps}]: ")
    if not parameters['num_inference_steps']:
        parameters['num_inference_steps'] = default_num_inference_steps
    parameters['num_inference_steps'] = int(parameters['num_inference_steps'])

    seed = "-1"
    parameters['seeds']=[]

    print("Enter the seeds for the images to be generated.")
    print("The number of seeds equals the number of generated images.")
    print("Press enter when no more seeds should be added.")
    while seed != "":
        seed = input("Seed: ")
        if seed != "":
            parameters['seeds'].append(int(seed))

    parameters['guidance_scale'] = input(
        f"guidance_scale (1-20) [{default_guidance_scale}]: ")
    if not parameters['guidance_scale']:
        parameters['guidance_scale'] = default_guidance_scale
    parameters['guidance_scale'] = float(parameters['guidance_scale'])
    parameters['out_path']=""

def mk_out_dir():
    global parameters
    
    if not parameters['out_path'] or os.path.exists(parameters['out_path']):
        parameters['out_path'] = f"./{int(time.time())}"
    os.makedirs(parameters['out_path'])

def load_model():
    print(f"Loading Model {parameters['model_id']}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        parameters['model_id'],
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config)

    pipeline = pipeline.to(device)


def main():
    
    get_parameters()
    
    mk_out_dir()
    
    with open(f'{parameters['out_path']}/parameters.json', 'w') as params_file: 
        params_file.write(json.dumps(parameters))
    
    load_model()

    print(f"Generating images in {parameters['out_path']} with parameters: {parameters}")
    
    images = pipeline(callback_on_step_end=callback, **parameters_to_pipeline_args(parameters)).images
    i = 0
    for image in images:
        image.save(f"{parameters['out_path']}/{parameters['seeds'][i]}.png")
        i += 1

    print(f"Saved output images to {parameters['out_path']}")


if __name__ == "__main__":
    main()
