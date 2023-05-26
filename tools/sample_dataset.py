import argparse
import os

import pandas as pd
import random
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmt_path', type=str, default='./work_dirs/dataset/{}')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--rate', type=float, default=1 / 6)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--sample', type=str, default='class', choices=['stf', 'class'])
    args = parser.parse_args()
    args.txt_path = os.path.join(args.fmt_path.format(args.dataset), 'label.txt')
    return args


args = get_args()
random.seed(args.seed)
np.random.seed(args.seed)

dict_ = {'path': [], 'label': []}
with open(args.txt_path, 'r') as f:
    for line in f:
        path, label = line.split()[0], int(line.split()[1])
        dict_['path'].append(path)
        dict_['label'].append(label)

df = pd.DataFrame(dict_)

if args.sample == 'stf':
    min_n_label = df['label'].value_counts().min()
    sample_df = df.groupby('label').apply(lambda x: x.sample(frac=args.rate))
    sample_df = sample_df.drop(sample_df[sample_df['label'] < min_n_label].index)
elif args.sample == 'class':
    total_label = df['label'].unique()
    n_label = len(total_label)
    sample_label = np.random.choice(total_label, int(n_label * args.rate), replace=False)
    sample_df = df[df['label'].isin(sample_label)]
else:
    raise NotImplementedError

str_lines = ['{} {}\n'.format(p, l) for p, l in zip(sample_df['path'], sample_df['label'])]
str_lines[-1] = str_lines[-1][:-1]  # drop last \n

out_dir, _ = os.path.split(args.txt_path)
out_path = os.path.join(out_dir, '{}_sample_label.txt'.format(args.sample))
print(out_path)
with open(out_path, 'w') as f:
    f.writelines(str_lines)
