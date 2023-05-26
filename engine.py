import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda import amp
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from eval_utils import eval_pair
import sklearn
import os

from data import train_dataset, val_dataset
import torch_utils as tu

import heads

from models import iresnet
from models import mobilefacenet

test_trans = A.Compose([
    A.Resize(112, 112),
    A.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
train_trans = A.Compose([
    A.Resize(112, 112),
    A.HorizontalFlip(),
    A.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


def build_dataloader(args):
    print('training on {} dataset'.format(args.train_root_dir.split('/')[-1]))
    print('using txt file: {}'.format(os.path.split(args.train_pair_txt)[-1]))

    train_dataset_ = train_dataset.TXTPairDataset(args.train_root_dir, args.train_pair_txt,
                                                  transform=lambda x: train_trans(image=x)['image'], to_array=True)
    val_all_dataset = val_dataset.ValDataset(args.val_root_dir,
                                             transform=lambda x: test_trans(image=x)['image'], to_array=True)

    train_sampler, val_sampler = None, None
    if args.distributed:
        train_sampler = data.DistributedSampler(train_dataset_, seed=args.seed)
        train_batch_sampler = data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
        train_dataloader = data.DataLoader(train_dataset_, batch_sampler=train_batch_sampler,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory)

        val_sampler = data.DistributedSampler(val_all_dataset, seed=args.seed)
        val_dataloader = data.DataLoader(val_all_dataset, args.batch_size // 2, sampler=val_sampler,
                                         num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    else:
        train_dataloader = data.DataLoader(train_dataset_, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory,
                                           worker_init_fn=tu.model_tool.seed_worker)
        val_dataloader = data.DataLoader(val_all_dataset, batch_size=args.batch_size // 2, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=args.pin_memory,
                                         worker_init_fn=tu.model_tool.seed_worker)

    return train_dataloader, val_dataloader, train_sampler, val_sampler


def build_model(args):
    if args.model == 'iresnet18':
        model = iresnet.iresnet18(fp16=args.use_16bit)
    elif args.model == 'iresnet50':
        model = iresnet.iresnet50(fp16=args.use_16bit)
    elif args.model == 'iresnet100':
        model = iresnet.iresnet100(fp16=args.use_16bit)
    elif args.model == 'mbf':
        model = mobilefacenet.get_mbf(fp16=args.use_16bit)
    elif args.model == 'mbf_large':
        model = mobilefacenet.get_mbf_large(fp16=args.use_16bit)
    else:
        raise NotImplementedError

    if tu.dist.is_dist_avail_and_initialized() and args.ddp_sync:
        # switch batchnorm to syncbathcnorm
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


def build_head(args, in_features, out_features):
    if args.head_type == 'CentersBiasFace':
        head = heads.CentersBiasFace(in_features, out_features,
                                   s=args.scale, m=args.margin, m_m=args.m_m,
                                   constant_t=args.constant_t)
    else:
        raise NotImplementedError
    return head


def ddp_module_replace(param_ckpt):
    return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}


def resume_ckpt(args, backbone, head, optimizer, lr_optimizer, ddp=False):
    epoch = 0
    if args.resume_dir:
        resume_ckpts = [f for f in os.listdir(args.resume_dir) if f.startswith('latest')]
        assert len(resume_ckpts) == 1
        resume_name = resume_ckpts[0]
        ckpt = torch.load(os.path.join(args.resume_dir, resume_name), map_location='cpu')

        backbone_ckpt = ckpt['backbone'] if ddp else ddp_module_replace(ckpt['backbone'])
        head_ckpt = ckpt['head'] if ddp else ddp_module_replace(ckpt['head'])

        backbone.load_state_dict(backbone_ckpt)
        if head.state_dict():
            head.load_state_dict(head_ckpt)
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_optimizer.load_state_dict(ckpt['lr_optimizer'])
        epoch = ckpt['epoch'] + 1
    return epoch


def train_one_epoch(args,
                    model: nn.Module,
                    head: nn.Module,
                    dataloader,
                    optimizer,
                    save_manager: tu.save.SaveManager,
                    grad_scaler,
                    epoch,
                    max_iter=float('inf')):
    model.train()
    head.train()

    metric_logger = tu.metric.MetricMeterLogger()

    for index, unpacked in enumerate(
            metric_logger.log_every(dataloader, header='train', desc=f'train epoch {epoch}')
    ):
        if index > max_iter:
            break

        images, labels = unpacked
        images, labels = images.cuda(), labels.cuda()

        embeddings = model(images)

        loss_dict = head(embeddings, labels, epoch=epoch)
        loss = loss_dict['total_loss']
        metric_logger.update(**loss_dict)

        if args.use_16bit:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        step_logs = metric_logger.values()
        step_logs['epoch'] = epoch
        save_manager.save_step_log('train', **step_logs)

    epoch_logs = metric_logger.get_finish_epoch_logs(force=True)
    epoch_logs['epoch'] = epoch
    save_manager.save_epoch_log('train', **epoch_logs)

    return epoch_logs


@torch.no_grad()
def val_one_epoch(args,
                  model: nn.Module,
                  dataloader,
                  save_manager: tu.save.SaveManager,
                  epoch,
                  flip=False,
                  max_iter=float('inf')):
    model.eval()

    _iter = tqdm(iter(dataloader), desc=f'val epoch {epoch}') if tu.dist.is_master_process() else dataloader

    pos_emb_list = []
    neg_emb_list = []
    label_list = []
    idx_list = []

    for dataset_index, (images, labels, dataset_idx) in enumerate(_iter):
        if dataset_index > max_iter:
            break

        for i in range(2):
            if i == 0:
                image1, image2 = images
                image1, image2 = image1.cuda(), image2.cuda()
            if i == 1:
                if flip:
                    image1, image2 = torch.flip(image1, dims=[3]), torch.flip(image2, dims=[3])
                else:
                    break

            emb1 = model(image1)
            emb2 = model(image2)

            pos_emb_list.append(emb1.detach().cpu().numpy())
            neg_emb_list.append(emb2.detach().cpu().numpy())

            label_list.append(labels.cpu().numpy())
            idx_list.append(dataset_idx.cpu().numpy())
    print('evaluate gathering')
    all_list = tu.dist.all_gather((pos_emb_list, neg_emb_list, label_list, idx_list))

    log_metric = average_acc = 0.

    if tu.dist.is_master_process():
        # only evaluate on master process, alleviate the cpu
        label_list = []
        idx_list = []

        pos_emb_list = []
        neg_emb_list = []

        for sub_list in all_list:
            pos_emb, neg_emb, label, idx = sub_list
            pos_emb_list.extend(pos_emb)
            neg_emb_list.extend(neg_emb)
            label_list.extend(label)
            idx_list.extend(idx)

        pos_embeddings = np.concatenate(pos_emb_list, axis=0)
        neg_embeddings = np.concatenate(neg_emb_list, axis=0)
        is_sames = np.concatenate(label_list, axis=0)
        dataset_indexes = np.concatenate(idx_list, axis=0)

        unique_indexes = np.unique(dataset_indexes)
        unique_indexes.sort()

        log_dict = {}

        for dataset_index in unique_indexes:
            indices = np.where(dataset_indexes == dataset_index)

            dataset_name = dataloader.dataset.get_dataset_name(dataset_index)
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

            sub_dict = {f"{dataset_name}_AccMean": accuracy.mean(), f"{dataset_name}_AccStd": accuracy.std()}
            average_acc += accuracy.mean().item()

            print(sub_dict)
            for k, v in sub_dict.items():
                log_dict[k] = v

        average_acc /= len(unique_indexes)

        log_dict['epoch'] = epoch
        save_manager.save_epoch_log('val', **log_dict)

        log_metric = average_acc
    tu.dist.barrier()

    return log_metric
