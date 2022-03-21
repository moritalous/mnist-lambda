import json

import base64
import io
import os
import re
import warnings

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

from PIL import Image as pil_image

_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}

model = load_model('mnist_model.h5')


def load_img(image_binary, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest', keep_aspect_ratio=False):

    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
     
    img = pil_image.open(io.BytesIO(image_binary))

    if color_mode == 'grayscale':
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ('L', 'I;16', 'I'):
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart, crop_box_hstart, crop_box_wend,
                    crop_box_hend
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img


def inference(image_base64):
    image_binary = base64.b64decode(image_base64)
    img= img_to_array(load_img(image_binary, target_size=(28, 28), color_mode = 'grayscale'))
    
    X = []
    X.append(img)

    X = np.asarray(X)
    X = X.astype('float32')
    X = X / 255.0

    features = model.predict(X)
    return {
        'inference': int(features.argmax())
    }

def lambda_handler(event, context):

    result = inference(event['body'])

    print(result)

    return {
        "statusCode": 200,
        "body": json.dumps(result),
    }
