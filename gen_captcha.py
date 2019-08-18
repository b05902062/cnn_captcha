# -*- coding:utf-8 -*-
import argparse
import json
import string
import os
import shutil
import numpy as np
import uuid
from captcha.image import ImageCaptcha

FLAGS = None
META_FILENAME = 'meta.json'


def get_choices():
    choices = [
        (FLAGS.digit, map(str, range(10))),
        (FLAGS.lower, string.ascii_lowercase),
        (FLAGS.upper, string.ascii_uppercase),
        ]
    return tuple([i for is_selected, subset in choices for i in subset if is_selected])


def _gen_captcha(img_dir, num_per_image,sampleNum, width, height, choices):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    print('generating %s captchas in %s' % (sampleNum, img_dir))
    for a in range(sampleNum):
        i=[choices[np.random.randint(0,len(choices))] for dNum in range(num_per_image) ]
        captcha = ''.join(i)
        fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
        image.write(captcha, fn)


def build_file_path(x):
    return os.path.join(FLAGS.data_dir, 'char-%s-sample-%s' % (FLAGS.npi, FLAGS.s), x)


def gen_dataset():
    n_sample = FLAGS.s
    num_per_image = FLAGS.npi
    test_ratio = FLAGS.t

    choices = get_choices()
    if FLAGS.width:
        width=FLAGS.width
    else:
        width=40 + 20 * num_per_image

    if FLAGS.height:
        height=FLAGS.height
    else:
        height=100

    # meta info
    meta = {
        'num_per_image': num_per_image,
        'label_size': len(choices),
        'label_choices': ''.join(choices),
        'width': width,
        'height': height,
    }

    print('%s choices: %s' % (len(choices), ''.join(choices) or None))

    _gen_captcha(build_file_path('train'), num_per_image,n_sample, width, height, choices=choices)
    _gen_captcha(build_file_path('test'), num_per_image,int(n_sample*test_ratio), width, height, choices=choices)

    meta_filename = build_file_path(META_FILENAME)
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        default=1000,
        type=int,
        help='the number of captchas to generate for training.')
    parser.add_argument(
        '--height',
        type=int,
        help='number of pixel in height.')
    parser.add_argument(
        '--width',
        type=int,
        help='number of pixel in width.')
    parser.add_argument(
        '-t',
        default=0.2,
        type=float,
        help='ratio of test dataset. default to 0.2. -s * -t captchas will be generated for testing')

    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '--data_dir',
        default='./images',
        type=str,
        help='where data will be saved.')

    FLAGS, unparsed = parser.parse_known_args()
    if not (FLAGS.digit or FLAGS.upper or FLAGS.lower):
        parser.error('specify at least one among -d -l -u.')	

    gen_dataset()
