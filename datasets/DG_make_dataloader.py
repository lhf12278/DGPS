import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from prettytable import PrettyTable
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .cuhk_sysu import CUHK_SYSU
from .dukemtmcreid import DukeMTMCreID
from .dukemtmcreid_sct import DukeMTMCreID_SCT
from .market1501_sct import Market1501_SCT
from .market1501 import Market1501
from .msmt17 import MSMT17
from .msmt17_sct import MSMT17_SCT
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .cuhk03 import cuhk03_np
__factory = {
    'market1501': Market1501,
    'market1501_sct': Market1501_SCT,
    'dukemtmc': DukeMTMCreID,
    'dukemtmc_sct': DukeMTMCreID_SCT,
    'msmt17': MSMT17,
    'msmt17_sct':MSMT17_SCT,
    'occ_duke': OCC_DukeMTMCreID,
    'cuhk03':cuhk03_np,
    'cuhk_sysu':CUHK_SYSU,
    'veri': VeRi,
    'VehicleID': VehicleID,
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def combine_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples = []
    max_pid, max_cid, max_sid = 0,0,0
    t = 0
    for samples in samples_list:
        for a_sample in samples:
            if t == 0:
                img_path = a_sample[0]
                pid = max_pid + a_sample[1]
                cid = max_cid + a_sample[2]
                sid = max_sid + a_sample[3]
                all_samples.append([img_path, pid, cid,sid])
            else:
                img_path = a_sample[0]
                pid = max_pid + a_sample[1]+1
                cid = max_cid + a_sample[2]+1
                sid = max_sid + a_sample[3]+1
                all_samples.append([img_path, pid, cid, sid])
        t = t+1
        max_pid = max([sample[1] for sample in all_samples])
        max_cid = max([sample[2] for sample in all_samples])
        max_sid = max([sample[3] for sample in all_samples])
    return all_samples


def relabels(samples, label_index1, label_index2, label_index3):
    '''
    根据在label_index1、label_index2和label_index3处找到的唯一值重新排序标签。
    '''
    # 创建字典以将旧标签映射到新标签
    label_mapping1 = {}
    label_mapping2 = {}
    label_mapping3 = {}

    # 遍历样本以构建标签映射
    for sample in samples:
        label1 = sample[label_index1]
        label2 = sample[label_index2]
        label3 = sample[label_index3]

        if label1 not in label_mapping1:
            label_mapping1[label1] = len(label_mapping1)
        if label2 not in label_mapping2:
            label_mapping2[label2] = len(label_mapping2)
        if label3 not in label_mapping3:
            label_mapping3[label3] = len(label_mapping3)

    # 创建一个新的样本列表，用于存储修改后的样本
    new_samples = []

    # 使用映射更新每个样本中的标签，并将其添加到新的样本列表中
    for sample in samples:
        new_sample = sample.copy()  # 创建一个副本以防止修改原始样本
        new_sample[label_index1] = label_mapping1[sample[label_index1]]
        new_sample[label_index2] = label_mapping2[sample[label_index2]]
        new_sample[label_index3] = label_mapping3[sample[label_index3]]
        new_samples.append(new_sample)

    return new_samples

def show_info(self, train, query, gallery, name=None):
    def analyze(samples):
        pid_num = len(set([sample[1] for sample in samples]))
        cid_num = len(set([sample[2] for sample in samples]))
        sid_num = len(set([sample[3] for sample in samples]))
        sample_num = len(samples)
        return sample_num, pid_num, cid_num,sid_num

    train_info = analyze(train)
    query_info = analyze(query)
    gallery_info = analyze(gallery)

    # please kindly install prettytable: ```pip install prettyrable```
    table = PrettyTable(['set', 'images', 'identities', 'cameras', 'sources'])
    table.add_row([self.__class__.__name__ if name is None else name, '', '', '',''])
    table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2]),str(train_info[3])])
    table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2]),str(query_info[3])])
    table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2]),str(gallery_info[3])])
    print(table)
    pids_num = train_info[1]
    cids_num = train_info[2]
    sid_num = train_info[3]

    return pids_num,cids_num,sid_num


def Dg_make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    samples_list_train = []
    samples_list_query = []
    samples_list_gallery = []

    if len(cfg.DATASETS.NAMES) > 1:
        for data_name in cfg.DATASETS.NAMES:
            dataset = __factory[data_name](root=cfg.DATASETS.ROOT_DIR)
            samples_list_train.append(dataset.train)
            samples_list_query.append(dataset.query)
            samples_list_gallery.append(dataset.gallery)
        samples_train = combine_samples(samples_list_train)
        samples_query = combine_samples(samples_list_query)
        samples_gallery = combine_samples(samples_list_gallery)
        dataset_train = relabels(samples_train, 1, 2, 3)
        dataset_query = relabels(samples_query, 1, 2, 3)
        dataset_gallery = relabels(samples_gallery, 1, 2, 3)
        pids_num, cids_num, sids_num = show_info(None, dataset_train, dataset_query, dataset_gallery,name=str(cfg.DATASETS.NAMES))
        train_set = ImageDataset(dataset_train, train_transforms)
        train_set_normal = ImageDataset(dataset_train, val_transforms)
        num_classes = pids_num
        cam_num = cids_num
        view_num = sids_num


    else:
        dataset = __factory[cfg.DATASETS.NAMES[0]](root=cfg.DATASETS.ROOT_DIR)
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            if len(cfg.DATASETS.NAMES)>1:
                dataset_train = dataset_train
            else:
                dataset_train = dataset.train
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset_train, cfg.SOLVER.IMS_PER_BATCH,
                                                     cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            if len(cfg.DATASETS.NAMES)>1:
                dataset_train = dataset_train
            else:
                dataset_train = dataset.train
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, drop_last=False
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_data = {}
    val_query_num = {}
    for data in cfg.DATASETS.NAMES_TARGET:
        dataset_target = __factory[data](root=cfg.DATASETS.ROOT_DIR)
        val_set = ImageDataset(dataset_target.query + dataset_target.gallery, val_transforms)
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        val_data[data] = val_loader
        val_query_num[data] = len(dataset_target.query)


    train_loader_normal = DataLoader(
                train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    return train_loader, train_loader_normal, val_data, val_query_num, num_classes, cam_num, view_num
