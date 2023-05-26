import engine
import torch
from torch.cuda import amp
import os

import torch_utils as tu
from torch_utils.config import get_args, create_output_dir


def main(args):
    tu.dist.init_distributed_mode(args)
    seed = tu.dist.get_rank() + args.seed
    tu.model_tool.seed_everything(seed)

    if not tu.dist.is_dist_avail_and_initialized():
        torch.multiprocessing.set_sharing_strategy('file_system')

    create_output_dir(args)
    print('output dir: {}'.format(args.output_dir))

    train_dataloader, val_dataloader, train_sampler, val_sampler = engine.build_dataloader(args)

    train_num_classes = train_dataloader.dataset.num_classes

    print('model: {}'.format(args.model))
    model = engine.build_model(args)

    print('head: {}'.format(args.head_type))
    head = engine.build_head(args, 512, train_num_classes)

    # set ddp mode
    model.cuda()
    model = tu.dist.ddp_model(model, [args.local_rank])

    head.cuda()
    head = tu.dist.ddp_model(head, [args.local_rank])

    parameters = [{'params': model.parameters()}, {'params': head.parameters()}]
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_optimizer = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.gamma)

    print('finish loading model')

    if args.use_16bit:
        print('using amp mode')
    grad_scaler = amp.GradScaler(enabled=args.use_16bit)

    # resume checkpoint
    start_epoch = engine.resume_ckpt(args, model, head, optimizer, lr_optimizer, ddp=True)
    if start_epoch != 0:
        print('resume from checkpoint')

    save_manager = tu.save.SaveManager(args.output_dir, args.model, 'acc', ckpt_save_freq=args.ckpt_save_freq)
    save_manager.save_hparam(args)

    for epoch in range(start_epoch, args.end_epoch):
        if tu.dist.is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        tu.dist.barrier()

        engine.train_one_epoch(args, model, head, train_dataloader, optimizer, save_manager, grad_scaler, epoch)

        lr_optimizer.step()

        log_acc = engine.val_one_epoch(args, model, val_dataloader, save_manager, epoch, flip=False)
        ckpt = {
            'backbone': model.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_optimizer': lr_optimizer.state_dict(),
            'epoch': epoch
        }
        save_manager.save_ckpt(ckpt, ckpt['backbone'], epoch, log_acc)

    torch.cuda.empty_cache()


def add_config(parser):
    # model settings
    parser.add_argument('--model', type=str)

    # head settings
    parser.add_argument('--head_type', type=str)

    # ada_margin_face settings
    parser.add_argument('--scale', type=float, default=64.0)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--m_m', type=float, default=0.05)

    # dataset settings
    parser.add_argument('--fmt_path', type=str, default='./work_dirs/dataset/{}')
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--sample', type=str, default='', choices=['stf', 'class', ''])
    parser.add_argument('--val_root_dir', type=str, default='./work_dirs/dataset/ms1mv2/val')

    # learning settings
    parser.add_argument('--lr_milestones', type=str, default='8,12,16')

    # save settings
    parser.add_argument('--ckpt_save_freq', type=int, default=1)

    parser.add_argument('--ddp_sync', action='store_true')
    parser.add_argument('--constant_t', action='store_true')
    return parser


def preprocess_args(args):
    args.lr_milestones = list(map(int, args.lr_milestones.split(',')))
    args.pin_memory = False

    assert args.dataset
    args.train_root_dir = args.fmt_path.format(args.dataset)

    if args.sample:
        args.use_sample = True
        train_pair_txt = os.path.join(args.train_root_dir, '{}_sample_label.txt'.format(args.sample))
        assert os.path.exists(train_pair_txt)
        args.train_pair_txt = train_pair_txt
    else:
        except_set = {'glintmini'}
        if args.dataset not in except_set:
            args.train_pair_txt = os.path.join(args.train_root_dir, 'label.txt')
        elif args.dataset == 'glintmini':
            args.train_pair_txt = os.path.join(args.train_root_dir, 'Glint-Mini.list')
        else:
            raise NotImplementedError

    return args


if __name__ == '__main__':
    args = get_args(add_config)
    args = preprocess_args(args)
    main(args)
