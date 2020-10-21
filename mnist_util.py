import struct
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import urllib
import os
import gzip

def download_data():
    if not os.path.exists('mnist'):
        os.mkdir('mnist')
    fnames = [
        'train-labels-idx1-ubyte',
        'train-images-idx3-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
    ]
    for fname in fnames:
        if not os.path.exists(f'mnist/{fname}'):
            if not os.path.exists(f'mnist/{fname}.gz'):
                print(f'http://yann.lecun.com/exdb/mnist/{fname}.gz')
                urllib.request.urlretrieve(f'http://yann.lecun.com/exdb/mnist/{fname}.gz',
                                           f'mnist/{fname}.gz')
            with open(f'mnist/{fname}', 'wb') as f:
                f.write(gzip.open(f'mnist/{fname}.gz', 'rb').read())


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

def labeled_image_array(images, labels, colors=None, max_width=800):
    n, im_h, im_w = images.shape

    try:
        font = ImageFont.truetype(r'arial.ttf', 14)
    except:
        font = ImageFont.truetype(r'DejaVuSans.ttf', 14)

    uq_labels = set(l for lab in labels for l in lab)
    lbl_w = max(font.getsize(l)[0] for l in uq_labels)
    textheight = sum(font.getmetrics())
    box_w = 4 + max(im_w, lbl_w)
    box_h = 4 + im_h + textheight*len(labels)
    columns = min(n, max(1, max_width // box_w))
    rows = (n + columns - 1) // columns
    im = Image.new('RGB', (columns*box_w, rows*box_h), (255,255,255))
    draw = ImageDraw.Draw(im)
    for i, ims_data in enumerate(images):
        ims = Image.fromarray(ims_data, 'L')
        x = i % columns
        y = i // columns
        im.paste(ims, (2+x*box_w+(box_w-im_w)//2, 2+y*box_h))
        for k, lbl in enumerate(labels):
            if colors is not None:
                fg, bg = colors[k][i]
                bbox = draw.textbbox((2+x*box_w+box_w//2, 4+y*box_h+im_h+k*textheight), lbl[i], font=font, anchor='mt')
                draw.rectangle(bbox, fill=bg)
            else:
                fg = 0
            draw.text((2+x*box_w+box_w//2, 4+y*box_h+im_h+k*textheight), lbl[i], font=font, anchor='mt', fill=fg)
    return im
