# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import val_collate_fn
from .datasets import init_dataset, ImageDataset
from .transforms import build_transforms


def make_gallery_loader(cfg):
    val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return gallery_loader

def make_query_loader(cfg):
    val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    query_set = ImageDataset(dataset.query, val_transforms)
    query_loader = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return query_loader