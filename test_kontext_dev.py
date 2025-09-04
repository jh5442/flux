import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from itertools import product
import os

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

save_image_dir = "/home/ubuntu/jin-Vol/results/flux_kontext_dev"

img_paths = ["/home/ubuntu/jin-Vol/data/flux_kontext/city.png",
             "/home/ubuntu/jin-Vol/data/flux_kontext/rome.png",
             "/home/ubuntu/jin-Vol/data/flux_kontext/forest.png",
             "/home/ubuntu/jin-Vol/data/flux_kontext/ranch.png"]

edit_prompts = {"snow": "Make it snow-covered.",
                "cloudy": "Make it cloudy.",
                "cloudy-gloomy": "Make it cloudy and gloomy.",
                "rainy": "Make it rainy, and we can see puddles on the ground.",
                "sunset": "Change it to the sunset time of the day.",
                "night": "Change it to the night time of the day. Lights are on in the buildings and there are lamps on the street.",
                "explosion": "Make the building explode and it is on fire.",
                "neon": "Change it to the neon time of the day. There is neon light everywhere.",
                "van-gogh": "Change it to van-gogh styled.",
                "foggy": "Make it foggy.",
                "anime": "Make it anime styled.",
                "watercolor": "Change it to watercolor styled.",
                "sketch": "Make it sketch/graffiti styled.",
                "pixel": "Make it pixel-art styled.",
                "angle": "Change the angle of the view to a bird-view.", }


# edit_prompts = {"angle": "Change the angle of the view. Imagine the camera is high-up in the sky and looking down.", }

effects = edit_prompts.keys()

for combo in product(img_paths, effects):
    image_path = combo[0]
    one_effect_prompt = edit_prompts[combo[1]]

    input_image = load_image(image_path)

    image = pipe(image=input_image,
                 prompt=one_effect_prompt,
                 height=720,
                 width=1280,
                 guidance_scale=2.5).images[0]

    save_result_path = os.path.join(save_image_dir, image_path.split("/")[-1].split(".")[0] + "_" + combo[1] + ".png")
    image.save(save_result_path)
    print("Image saved to {}".format(save_result_path))
