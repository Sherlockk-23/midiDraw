import os
import sys

# from draw2pic import draw2pic
from .quickdraw_dataset.binary_parser import unpack_drawings, vector_to_raster, filter_images
from .midi_dataset.midi2img import img2midi
import numpy as np
import random
# import matplotlib.pyplot as plt

def get_filtered_dataset(root_path='./quickdraw_dataset', drop_rate = 0.03):

    chosen_classes = ['key', 'guitar', 'face', 'eye', 'eyeglasses', 'pants']

    vector_images = []
    cnt = 0
    for class_name in chosen_classes:
        for drawing in unpack_drawings(f'{root_path}/full_binary_{class_name}.bin'):
            if random.random() > drop_rate:
                continue
            vector_images.append(drawing['image'])
            cnt+=1
            # if cnt % 100 == 0:
            #     print(cnt)
                
    print(f"Total vector images: {len(vector_images)}")

    raster_images = vector_to_raster(vector_images, side=100, line_diameter=16, padding=16)

    print(f"Total raster images: {len(raster_images)}")
    print(f"Raster image shape: {raster_images[0].shape}")
    print(f"min pixel value: {np.min(raster_images)}, max pixel value: {np.max(raster_images)}")
    filtered_images = filter_images(raster_images, sum_interval=[1000, 4000])
    print(f"Filtered images: {len(filtered_images)}")
    return filtered_images

# filtered_images = filtered_images[:10]

if __name__ == "__main__":
    midis = []

    output_dir = './img_midis'

    filtered_images = get_filtered_dataset()

    for i, img in enumerate(filtered_images):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(filtered_images)}")
        img = img.reshape(100, 100, 1)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        midi = img2midi(img, min_duration_unit=0.1136, pad=6)
        try:
            midi = img2midi(img, min_duration_unit=0.1136, pad=6)
            midi.write(f"{output_dir}/{i}.mid")
        except Exception as e:
            print(f"Error processing image {i}: {e}")