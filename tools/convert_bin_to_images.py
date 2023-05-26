import argparse
from glob import glob
import os
import cv2
import mxnet as mx
from mxnet import ndarray as nd
import pickle
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def load_bin_to_torch(path, image_size):
    # load bins from face emorce dataset to torch.tensor
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list


def convert_bin_to_images(inbin_path, output_dir, image_size):
    try:
        with open(inbin_path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(inbin_path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3

    dataset = np.empty(((len(issame_list) * 2), image_size[0], image_size[1], 3))

    for idx in tqdm(range(len(issame_list) * 2), desc='decoding bin'):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])

        # BGR to RGB
        dataset[idx][:] = img.asnumpy()[..., ::-1]
    # print(len(dataset))

    output_image_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_image_dir, exist_ok=True)

    pair_filename = 'images/{}.jpg'
    filename = '{}.jpg'

    pair_fp = open(os.path.join(output_dir, 'pair.list'), 'w')
    image_fp = open(os.path.join(output_dir, 'image.list'), 'w')

    for index in tqdm(range(len(issame_list)), desc='creating pairs'):
        img1 = dataset[index * 2]
        img2 = dataset[index * 2 + 1]

        cv2.imwrite(os.path.join(output_image_dir, filename.format(index * 2)), img1,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(output_image_dir, filename.format(index * 2 + 1)), img2,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        pair_fp.write("{}\t{}\t{}\n".format(pair_filename.format(index * 2), pair_filename.format(index * 2 + 1),
                                            int(issame_list[index])))

        image_fp.write("{}\n".format(pair_filename.format(index * 2)))
        image_fp.write("{}\n".format(pair_filename.format(index * 2 + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    val_targets = glob(os.path.join(args.root_dir, '*.bin'))
    val_targets = [target.split('/')[-1].split('.')[0] for target in val_targets]
    print('val targets : {}'.format(val_targets))

    for name in val_targets:
        print('converting {}'.format(name))

        output_dir = os.path.join(args.output_dir, name)

        os.makedirs(output_dir, exist_ok=True)

        convert_bin_to_images(os.path.join(args.root_dir, f"{name}.bin"), output_dir, (112, 112))
