import struct
import numpy as np
from PIL import Image

def read_labels(fname):
    with open(fname, 'rb') as f:
        magic, count = struct.unpack('>ll', f.read(8))
        assert magic == 2049
        data = f.read(count)
        assert len(data) == count
    return np.frombuffer(data, dtype=np.uint8, count=count)

def read_images(fname):
    with open(fname, 'rb') as f:
        magic, count, im_height, im_width = struct.unpack('>llll', f.read(16))
        assert magic == 2051
        bufsize = count*im_height*im_width
        data = f.read(bufsize)
        assert len(data) == bufsize
    return np.frombuffer(data, dtype=np.uint8, count=bufsize).reshape(count, im_width*im_height)

def image_matrix(im_array):
    h, w, im_h, im_w = im_array.shape
    im = Image.new('L', (w*(4+im_w), h*(4+im_h)), 128)
    for y in range(h):
        for x in range(w):
            ims = Image.fromarray(im_array[y][x], 'L')
            im.paste(ims, (2+x*(4+im_w), 2+y*(4+im_h)))
    return im
