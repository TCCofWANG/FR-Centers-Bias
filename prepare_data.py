import numpy as np
from sklearn import preprocessing
import torch
import engine
import os
import shutil
from data import train_dataset
from torch_utils.config import get_args
import torch_utils as tu
from tqdm import tqdm
from torch.utils import data


def load_model(model_path, model_name=None):
    try:
        model = torch.jit.load(model_path, map_location='cpu')
        print('teacher model: load jit')
    except RuntimeError or ValueError:
        if model_name is None:
            raise ValueError
        args.model = model_name
        model = engine.build_model(args)
        ckpt = torch.load(model_path, map_location='cpu')
        try:
            ckpt = ckpt['backbone']
        except KeyError:
            ckpt = ckpt
        ckpt = engine.ddp_module_replace(ckpt)
        model.load_state_dict(ckpt)
        print('teacher model: load pretrained state dict')
    model.eval()
    return model


def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()


def save_on_self_process(embeddings, labels, data_output):
    emb_output = os.path.join(data_output, 'embeddings_{}.npy'.format(tu.dist.get_rank()))
    label_output = os.path.join(data_output, 'labels_{}.npy'.format(tu.dist.get_rank()))

    embeddings = to_numpy(embeddings)
    labels = to_numpy(labels)

    np.save(emb_output, embeddings)
    np.save(label_output, labels)


@torch.no_grad()
def calc_embeddings(model, dataloader):
    _iter = tqdm(iter(dataloader), desc='calculating features')
    embedding_tensors = None
    label_tensors = None
    for index, (images, labels) in enumerate(_iter):

        images = images.cuda()

        embedding = model(images)

        embedding = embedding.detach().cpu()
        if embedding_tensors is None or label_tensors is None:
            embedding_tensors, label_tensors = embedding.detach(), labels
        else:
            embedding_tensors = torch.concat([embedding_tensors, embedding.detach()], dim=0)
            label_tensors = torch.concat([label_tensors, labels], dim=0)

    return embedding_tensors.cpu(), label_tensors.cpu()


def calc_info(embeddings, labels):
    assert len(embeddings) == len(labels)

    mean_centers = []
    std_centers = []
    label_centers = []
    norm_mean_centers = []

    unique_labels = np.unique(labels)
    unique_labels.sort()
    _iter = tqdm(unique_labels, desc='calculating centers')
    for label in _iter:
        label_centers.append(label)
        indexes = np.where(labels == label)
        mean_center = np.mean(embeddings[indexes], axis=0)
        mean_centers.append(mean_center)

        std_center = np.std(embeddings[indexes], axis=0)
        std_centers.append(std_center)

        norm_mean_center = np.mean(preprocessing.normalize(embeddings[indexes]), axis=0)
        norm_mean_centers.append(norm_mean_center)

    label_centers = np.array(label_centers)
    sort_indexes = np.argsort(label_centers)

    norm_mean_centers = np.array(norm_mean_centers)[sort_indexes]
    mean_centers = np.array(mean_centers)[sort_indexes]
    std_centers = np.array(std_centers)[sort_indexes]

    return mean_centers, std_centers, label_centers, norm_mean_centers


def calc_embeddings_phase(args):
    assert args.train_root_dir.split('/')[-1] == args.train_pair_txt.split('/')[-2]
    if args.train_root_dir.split('/')[-1] == 'ms1mv2':
        print('inf ms1mv2')
    else:
        print('inf ms1mv3')

    dataset = train_dataset.TXTPairDataset(args.train_root_dir, args.train_pair_txt,
                                           transform=lambda x: engine.test_trans(image=x)['image'],
                                           to_array=True)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False)

    model = load_model(args.teacher_path, args.teacher_model)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    check_path(args.data_output)
    print('calculate embeddings from teacher model')
    embedding_tensors, label_tensors = calc_embeddings(model, dataloader)

    model.cpu()
    del model
    torch.cuda.empty_cache()
    print('save embeddings and labels to disk')
    # save_on_self_process(embedding_tensors, label_tensors, args.data_output)

    emb_output = os.path.join(args.data_output, 'embeddings.npy')
    label_output = os.path.join(args.data_output, 'labels.npy')

    embeddings = to_numpy(embedding_tensors)
    labels = to_numpy(label_tensors)

    print('numbers of embeddings: {}'.format(len(embeddings)))
    print('numbers of labels: {}'.format(len(labels)))

    np.save(emb_output, embeddings)
    np.save(label_output, labels)


def calc_info_phase(args):
    emb_input = os.path.join(args.data_output, 'embeddings.npy')
    label_input = os.path.join(args.data_output, 'labels.npy')
    mean_output = os.path.join(args.data_output, 'mean_center.npy')
    std_output = os.path.join(args.data_output, 'std_center.npy')
    label_output = os.path.join(args.data_output, 'label_center.npy')
    norm_mean_output = os.path.join(args.data_output, 'norm_mean_center.npy')

    embeddings = np.load(emb_input)
    labels = np.load(label_input)

    mean_centers, std_centers, label_centers, norm_mean_centers = calc_info(embeddings, labels)

    np.save(mean_output, mean_centers)
    np.save(std_output, std_centers)
    np.save(label_output, label_centers)
    np.save(norm_mean_output, norm_mean_centers)


def split_embedding_to_files(args):
    embeddings = np.load(os.path.join(args.data_output, 'embeddings.npy'))
    emb_out_dir = os.path.join(args.data_output, 'embeddings')
    assert not os.path.exists(emb_out_dir)
    os.makedirs(emb_out_dir)
    pbar = tqdm(total=len(embeddings), desc='split features of numpy')
    for idx, emb in enumerate(embeddings):
        emb_tensor = torch.from_numpy(emb)
        torch.save(emb_tensor, os.path.join(emb_out_dir, '{}.pth'.format(idx)))
        pbar.update()


def prepare(args):
    tu.model_tool.seed_everything(args.seed)
    if args.mode == 0:
        calc_embeddings_phase(args)
    elif args.mode == 1:
        calc_info_phase(args)
    elif args.mode == 2:
        split_embedding_to_files(args)
    else:
        raise NotImplementedError


def add_config(parser):
    parser.add_argument('--teacher_model', type=str, default='iresnet100')
    parser.add_argument('--teacher_path', type=str, default='./experiments/benchmark/arcface_iresnet100_ms1mv3.pth')

    parser.add_argument('--fmt_path', type=str, default='./work_dirs/dataset/{}')
    parser.add_argument('--dataset', type=str, default='ms1mv2')
    parser.add_argument('--data_output', type=str, default='./experiments/predata')

    parser.add_argument('--mode', type=int)
    return parser


def preprocess_args(args):
    args.only_stu = False
    args.student_model = None

    args.train_root_dir = args.fmt_path.format(args.dataset)
    args.train_pair_txt = os.path.join(args.train_root_dir, 'label.txt')

    return args


if __name__ == '__main__':
    args = get_args(add_config)
    args = preprocess_args(args)
    prepare(args)
