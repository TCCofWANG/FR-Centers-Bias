import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val_file_path', type=str)
parser.add_argument('--round', type=int, default=2)
args = parser.parse_args()

with open(args.val_file_path, 'r') as f:
    val_dict = json.loads(f.readlines()[-1])

out_dict = {}
sum_value = 0.
dst_str = '_AccMean'
for k, v in val_dict.items():
    if k.endswith(dst_str):
        dataset_name = k[:-len(dst_str)]
        sum_value += v
        out_dict[dataset_name] = round(v * 100, args.round)
print(out_dict)
print('Mean: {}'.format(round(sum_value / len(out_dict) * 100, args.round)))
