import sys

sys.path.append('RMBG-1.4')

import os
import time
from multiprocessing import Pool
from typing import Optional

import torch
from briarmbg import BriaRMBG
from PIL import Image
from skimage import io
from utilities import postprocess_image, preprocess_image

net = BriaRMBG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
net.to(device)
net.eval()    

def remove_background(im_path) -> Optional[Image]:
    try:
        # prepare input
        model_input_size = [1024,1024]
        orig_im = io.imread(im_path)
        orig_im = orig_im[:,:,:3] # remove alpha channel
        orig_im_size = orig_im.shape[0:2]
        image = preprocess_image(orig_im, model_input_size).to(device)

        # inference 
        result=net(image)

        # post process
        result_image = postprocess_image(result[0][0], orig_im_size)

        # save result
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        orig_image = Image.open(im_path)
        no_bg_image.paste(orig_image, mask=pil_im)
        return no_bg_image

    except Exception as e:
        print(f"{e, im_path, orig_im.shape, orig_im_size}")
        return None


def process_image(im_path):
    try:
        no_bg_image = remove_background(im_path)
        if no_bg_image is None:
            print(f"Failed to remove background from {im_path}")
            return
        out_path = im_path.replace("data/pokemon", "data/pokemon_no_bg")
        out_path = f"{os.path.splitext(out_path)[0]}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        no_bg_image.save(out_path)
    except Exception as e:
        print(f"Error processing {im_path}: {e}")


def remove_background_multiprocess(image_paths, num_processes=4):
    with Pool(num_processes) as p:
        p.map(process_image, image_paths)


if __name__ == '__main__':
    # Benchmarking
    directory = 'data/pokemon/Pikachu'
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.splitext(filename)[1].lower() in [".jpg", ".png"]]

    start_time = time.time()
    remove_background_multiprocess(image_paths)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Background removal benchmark completed. Total execution time: {execution_time} seconds.")
