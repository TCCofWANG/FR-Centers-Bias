import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import numpy as np


class TXTPairDataset(Dataset):
    def __init__(self, root_dir, pairs_txt_path, transform=None, to_array=False):

        self.root_dir = root_dir
        self.img_label_txt_path = pairs_txt_path
        self.transform = transform

        self.to_array = to_array

        try:
            f = open(self.img_label_txt_path, 'r')
            self.img_label_txt = [[line.split()[0], int(line.split()[1])] for line in f]
        except FileNotFoundError as e:
            raise 'not found the label.txt file'

        classes = [item[1] for item in self.img_label_txt]
        classes_unique = list(set(classes))
        sort_info = torch.tensor(classes_unique).sort()
        map_dict = {k.item(): v.item() for k, v in zip(sort_info.values, sort_info.indices)}

        self.num_image = len(self.img_label_txt)
        self.num_classes = len(classes_unique)

        for i in range(self.num_image):
            self.img_label_txt[i][1] = map_dict[self.img_label_txt[i][1]]

        assert self.num_image == len(self.img_label_txt), 'number of images appear error'

    def parse_item(self, item):
        img_name, label = self.img_label_txt[item]
        img_file_path = os.path.join(self.root_dir, img_name)
        sample = Image.open(img_file_path)

        if self.to_array:
            sample = np.array(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label, img_file_path

    def __getitem__(self, item):
        sample, label, _ = self.parse_item(item)
        return sample, label

    def __len__(self):
        return self.num_image


class MS1MVx(TXTPairDataset):
    def __init__(self, root_dir, pairs_txt_path, transform=None, to_array=False):
        super().__init__(root_dir, pairs_txt_path, transform, to_array)

    def __getitem__(self, item):
        sample, label, img_file_path = self.parse_item(item)

        return sample, label


class MS1MV2(MS1MVx):
    def __init__(self, root_dir, pairs_txt_path, transform=None, to_array=False):
        super().__init__(root_dir, pairs_txt_path, transform, to_array)
        assert self.num_image == 5822653
        assert self.num_classes == 85742

    def __getitem__(self, item):
        return super().__getitem__(item)


class MS1MV3(MS1MVx):
    def __init__(self, root_dir, pairs_txt_path, transform=None, to_array=False):
        super().__init__(root_dir, pairs_txt_path, transform, to_array)
        assert self.num_image == 5179510
        assert self.num_classes == 93431

    def __getitem__(self, item):
        return super().__getitem__(item)


class GlintMini(TXTPairDataset):
    def __init__(self, root_dir, pairs_txt_path, transform=None, to_array=False):
        super().__init__(root_dir, pairs_txt_path, transform, to_array)


if __name__ == '__main__':
    dataset = 'ms1mv2'
    print(dataset)
    dataset = TXTPairDataset('../work_dirs/dataset/{}'.format(dataset),
                             '../work_dirs/dataset/{}/sample_label.txt'.format(dataset))
    print(dataset.__getitem__(0))
