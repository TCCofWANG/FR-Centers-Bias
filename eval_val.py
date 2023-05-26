import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
import engine
import argparse
from data import val_dataset
from tqdm import tqdm
import numpy as np
import sklearn
from eval_utils import eval_pair
import yaml
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_name', type=str, default='iresnet18')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--job', type=str, help='job name')
    parser.add_argument('--result_dir', type=str)
    return parser.parse_args()


def validate(args):
    # get val data
    val_all_dataset = val_dataset.ValDataset(args.val_root_dir,
                                             transform=lambda x: engine.test_trans(image=x)['image'],
                                             to_array=True)
    val_dataloader = DataLoader(val_all_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False,
                                drop_last=False, pin_memory=False)

    # get model to DP mode
    model = engine.build_model(args)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    # start to validate
    pos_emb_list = []
    neg_emb_list = []
    label_list = []
    idx_list = []

    _iter = tqdm(iter(val_dataloader), desc='validate val dataset')
    for images, labels, dataset_idx in _iter:
        for i in range(2):
            if i == 0:
                image1, image2 = images
                image1, image2 = image1.cuda(), image2.cuda()
            if i == 1:
                if args.flip:
                    image1, image2 = torch.flip(image1, dims=[3]), torch.flip(image2, dims=[3])
                else:
                    break

        if args.use_16bit:
            image1, image2 = image1.half(), image2.half()
        with amp.autocast(args.use_16bit):
            emb1 = model(image1).float()
            emb2 = model(image2).float()

        pos_emb_list.append(emb1.detach().cpu().numpy())
        neg_emb_list.append(emb2.detach().cpu().numpy())

        label_list.append(labels.cpu().numpy())
        idx_list.append(dataset_idx.cpu().numpy())

    pos_embeddings = np.concatenate(pos_emb_list, axis=0)
    neg_embeddings = np.concatenate(neg_emb_list, axis=0)
    is_sames = np.concatenate(label_list, axis=0)
    dataset_indexes = np.concatenate(idx_list, axis=0)

    unique_indexes = np.unique(dataset_indexes)
    unique_indexes.sort()

    log_dict = {}
    average_acc = 0.

    for dataset_index in unique_indexes:
        indices = np.where(dataset_indexes == dataset_index)

        dataset_name = val_all_dataset.get_dataset_name(dataset_index)
        print('evaluating {}'.format(dataset_name))

        is_sames_ = is_sames[indices]
        chosen_p_emb = pos_embeddings[indices]
        chosen_n_emb = neg_embeddings[indices]

        assert chosen_n_emb.shape[1] == chosen_p_emb.shape[1]
        embeddings_ = np.empty((len(chosen_p_emb) + len(chosen_n_emb), chosen_p_emb.shape[1]))
        embeddings_[0::2] = chosen_p_emb
        embeddings_[1::2] = chosen_n_emb
        embeddings_ = sklearn.preprocessing.normalize(embeddings_)

        tpr, fpr, accuracy, val, val_std, far = eval_pair.evaluate(embeddings_, is_sames_, dis='cosine')

        sub_dict = {'mean': accuracy.mean(), 'std': accuracy.std(), 'total': accuracy.tolist()}
        log_dict[dataset_name] = sub_dict
        average_acc += accuracy.mean().item()

        print(sub_dict)

    average_acc /= len(unique_indexes)
    log_dict['total avg'] = average_acc
    with open(os.path.join(args.result_dir, args.job + '.yaml'), 'w') as f:
        f.write(yaml.dump(log_dict))


if __name__ == '__main__':
    args = get_args()
    validate(args)
