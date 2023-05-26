import numpy as np
import torch.utils.data
from torch.utils.data import Dataset

import os
from PIL import Image


class LFW(Dataset):
    def __init__(self, root_dir, pairs_txt_path, transform=None, to_array=False) -> None:
        super(LFW, self).__init__()
        self.root_dir = root_dir
        self.pairs_txt_path = pairs_txt_path
        self.transform = transform
        self.to_array = to_array

        self.pair_label = []
        with open(self.pairs_txt_path, 'r') as f:
            for l in f.readlines()[1:]:
                class_index_pair = l.split()
                if len(class_index_pair) == 3:
                    # match
                    class_name, index_1, index_2 = class_index_pair
                    class_path = os.path.join(self.root_dir, class_name)

                    self.pair_label.append(
                        [class_path, int(index_1), class_path, int(index_2), 1])
                elif len(class_index_pair) == 4:
                    # mismatch
                    class_name_1, index_1, class_name_2, index_2 = class_index_pair
                    class_path_1 = os.path.join(self.root_dir, class_name_1)
                    class_path_2 = os.path.join(self.root_dir, class_name_2)

                    self.pair_label.append(
                        [class_path_1, int(index_1), class_path_2, int(index_2), 0])
                else:
                    raise ValueError()
        assert self.pair_label

        self.num_pairs = len(self.pair_label)

        self.class_to_idx = {}
        for idx, name in enumerate(os.listdir(root_dir)):
            self.class_to_idx[name] = idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, item):
        pair = self.pair_label[item]
        img_path_1, index_1, img_path_2, index_2, is_same = pair
        img_path_1, img_path_2 = self.get_path(img_path_1, index_1 - 1), \
            self.get_path(img_path_2, index_2 - 1)

        img_1, img_2 = Image.open(img_path_1), Image.open(img_path_2)

        if self.to_array:
            img_1, img_2 = np.array(img_1), np.array(img_2)

        if self.transform:
            img_1, img_2 = self.transform(img_1), self.transform(img_2)

        return [img_1, img_2], is_same
        # return img_1, label

    def __len__(self):
        return self.num_pairs

    @staticmethod
    def get_path(img_path, index):
        return os.path.join(img_path, os.listdir(img_path)[index])


class FacesBinDataset(Dataset):
    def __init__(self, root_dir, pairs_txt_path, dataset_idx, transform=None, to_array=False):
        self.root_dir = root_dir
        self.pairs_txt_path = pairs_txt_path
        self.dataset_idx = dataset_idx
        self.transform = transform
        self.to_array = to_array

        self.pairs = []
        img_path = os.path.join(self.root_dir, '{}')
        with open(self.pairs_txt_path, 'r') as f:
            for l in f.readlines():
                class_index_pair = l.strip().split()
                assert len(class_index_pair) == 3

                idx1, idx2, is_same = class_index_pair

                self.pairs.append((img_path.format(idx1), img_path.format(idx2), int(is_same)))

        assert self.pairs

        self.num_pairs = len(self.pairs)

    def __getitem__(self, item):
        pair = self.pairs[item]
        img_path1, img_path2, is_same = pair

        img1, img2 = Image.open(img_path1), Image.open(img_path2)

        if self.to_array:
            img1, img2 = np.array(img1), np.array(img2)

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)

        return [img1, img2], is_same, self.dataset_idx

    def __len__(self):
        return self.num_pairs


class ValDataset(Dataset):

    def __init__(self, root_dir, dataset_names=('agedb_30', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'lfw'),
                 transform=None, to_array=False):
        self.dataset_name_to_idx = {name: num for num, name in enumerate(dataset_names)}
        self.idx_to_dataset_name = {v: k for k, v in self.dataset_name_to_idx.items()}

        self.dataset_names = dataset_names

        dataset_root_dir = os.path.join(root_dir, '{}')
        dataset_pairs_path = os.path.join(dataset_root_dir, 'pair.list')

        datasets = [
            FacesBinDataset(dataset_root_dir.format(dataset_name), dataset_pairs_path.format(dataset_name),
                            self.dataset_name_to_idx[dataset_name],
                            transform=transform, to_array=to_array) for dataset_name in dataset_names
        ]
        self.dataset = torch.utils.data.ConcatDataset(datasets)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def get_dataset_name(self, idx):
        return self.idx_to_dataset_name[idx]
