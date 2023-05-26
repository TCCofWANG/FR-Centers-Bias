import argparse
import os
from time import strftime
from typing import Callable
from .dist import is_master_process
import yaml


def create_output_dir(args):
    args.output_dir = None
    if args.resume_dir:
        hparam_path = os.path.join(args.resume_dir, 'hparam.yaml')
        with open(hparam_path, 'r') as f:
            hparam_dict = yaml.load(f, yaml.FullLoader)
            args.output_dir = hparam_dict['output_dir']
    else:
        if args.save_log and is_master_process():
            current_time = strftime('%Y-%m-%d_%H-%M-%S')
            args.output_dir = os.path.join('work_dirs', args.prefix)

            current_exp = 0
            if os.path.exists(args.output_dir):
                exp_values = [int(f[3:]) for f in os.listdir(args.output_dir) if f.startswith('exp')]
                current_exp = max(exp_values) + 1 if exp_values else 0

            if args.exp_num != -1:
                current_exp = args.exp_num

            args.output_dir = os.path.join(args.output_dir, 'exp{}'.format(current_exp))

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            with open(os.path.join(args.output_dir, '{}.time'.format(current_time)), 'a+') as f:
                pass

            with open(os.path.join(args.output_dir, 'README'), 'a+') as f:
                pass

        else:
            args.output_dir = None

    return args


def base_config(parser):
    # Settings unrelated to hyperparameters
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--no_pin_memory', action='store_true')
    parser.add_argument('--prefix', type=str, default='default')

    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--exp_num', type=int, default=-1)

    return parser


def task_config(parser):
    # Settings related to training
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--end_epoch', type=int, default=50)
    parser.add_argument('--clip_max_norm', type=int, default=-1)

    parser.add_argument('--use_16bit', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--root_dir', type=str, default='')

    parser.add_argument('--ls', type=float, default=0.0)

    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    return parser


def get_args(add_config: Callable = None):
    parser = argparse.ArgumentParser()

    parser = base_config(parser)
    parser = task_config(parser)

    if add_config is not None:
        parser = add_config(parser)

    args = parser.parse_args()

    args.save_log = True if args.no_log is False else False
    args.pin_memory = True if args.no_pin_memory is False else False

    return args
