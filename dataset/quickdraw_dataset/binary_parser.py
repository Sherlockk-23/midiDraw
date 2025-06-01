import struct
from struct import unpack

import numpy as np
import matplotlib.pyplot as plt
import cairocffi as cairo


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def vector_to_raster(vector_images, side=88, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    cnt=0
    for vector_image in vector_images:
        
        # filter 
        if len(vector_image) <= 5:
            continue
        
        cnt+=1
        if cnt % 100 == 0:
            print(cnt, len(vector_images))
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images

def filter_images(raster_images, sum_interval = [1000, 4000]):
    """
    filter the images by the sum of the pixels
    """
    filtered_images = []
    for img in raster_images:
        if (img.sum()//255) > sum_interval[0] and (img.sum()//255) < sum_interval[1]:
            filtered_images.append(img)
    return filtered_images

# vector_images = []

# cnt = 0

# for drawing in unpack_drawings('full_binary_eye.bin'):
#     # do something with the drawing
#     # print(drawing['country_code'])
#     vector_images.append(drawing['image'])
#     cnt+=1
#     if cnt % 100 == 0:
#         print(cnt)
    
# # vector_images = vector_images[:10000]
# raster_images = vector_to_raster(vector_images, side=100, line_diameter=16, padding=16)
# # print(raster_images[0].shape)

# filtered_images = filter_images(raster_images, sum_interval=[1000, 4000])
# # filtered_images = raster_images
# print("Filtered images:", len(filtered_images))

# # plt.imshow(raster_images[0].reshape(100,100), cmap='gray')
# # plt.title("QuickDraw Visualization")
# # plt.show()
# # plt.savefig("quickdraw.png")

# filtered_images = np.array(filtered_images).reshape(-1, 100, 100).astype(np.uint8)

# # save the raster images to a file
# print("Saving raster images to file...")
# np.save('eye_100.npy', filtered_images)
# print("Saved.")

# # sample 100 imgs and show
# sample = filtered_images[:25]
# sample = np.array(sample)
# sample = sample.reshape(-1, 100, 100)
# print(sample.shape)
# for i in range(sample.shape[0]):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(sample[i], cmap='gray')
#     plt.title(f"Sum : {sample[i].sum()//255}")
#     plt.axis('off')
# # plt.tight_layout()
# plt.show()
# plt.savefig("quickdraw_sample.png")