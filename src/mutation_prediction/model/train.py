"""Train pooled attention against measured ddG or cached epsilon targets."""

import argparse
import os
from functools import partial

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from mutation_prediction.model.model import PooledAttentionModelPL
from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import (
    tied_featurize_mut,
)
from mutation_prediction.thermo_mpnn_d.thermompnn.parsers import get_v2_dataset


def parse_cfg(cfg):
    """
    Parse configuration scheme and set default arguments as needed
    """
    cfg.project = cfg.get('project', None)
    cfg.name = cfg.get('name', 'test')

    # data config
    cfg.data = cfg.get('data', {})
    cfg.data.mut_types = cfg.data.get('mut_types', ['single'])
    cfg.data.splits = cfg.data.get('splits', ['train', 'val'])
    cfg.data.side_chains = cfg.data.get('side_chains', False)
    cfg.data.refresh_every = cfg.data.get('refresh_every', 0)
    cfg.data.weight = cfg.data.get('weight', False)
    cfg.data.range = cfg.data.get('range', None)

    # training config
    cfg.training = cfg.get('training', {})
    cfg.training.num_workers = cfg.training.get('num_workers', 0)
    cfg.training.batch_size = cfg.training.get('batch_size', 256)
    cfg.training.epochs = cfg.training.get('epochs', 100)
    cfg.training.batch_fraction = cfg.training.get('batch_fraction', 1.0)
    cfg.training.shuffle = cfg.training.get('shuffle', True)

    cfg.training.learn_rate = cfg.training.get('learn_rate', 0.0001)
    cfg.training.mpnn_learn_rate = cfg.training.get('mpnn_learn_rate', None)
    cfg.training.lr_schedule = cfg.training.get('lr_schedule', True)

    # model config
    cfg.model = cfg.get('model', {})
    cfg.model.num_final_layers = cfg.model.get('num_final_layers', 2)
    cfg.model.freeze_weights = cfg.model.get('freeze_weights', True)
    cfg.model.load_pretrained = cfg.model.get('load_pretrained', True)
    cfg.model.mutant_embedding = cfg.model.get('mutant_embedding', False)
    
    # double mutant model options
    cfg.model.dropout = cfg.model.get('dropout', None)
    target_column = cfg.data.get('target_column', 'ddG_ML')
    cfg.model.target_name = cfg.model.get(
        'target_name', 'epsilon' if target_column == 'epsilon_target' else 'ddG'
    )

    return cfg


def train(cfg):
    cfg = parse_cfg(cfg)
    print('Configuration:\n', OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.training.get('seed', 42), workers=True)

    train_dataset, val_dataset = get_v2_dataset(cfg)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError('Training and validation splits must both be non-empty')

    collate = partial(tied_featurize_mut, side_chains=cfg.data.side_chains)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
    )

    model_pl = PooledAttentionModelPL(cfg)
    
    metric = cfg.training.get('checkpoint_metric', 'spearman')
    if metric not in ('mse', 'spearman'):
        raise ValueError('training.checkpoint_metric must be mse or spearman')
    monitor = f'val_{cfg.model.target_name}_{metric}'
    filename = cfg.name + '_{epoch:02d}_{' + monitor + ':.4f}'
    
    checkpath = os.path.abspath(cfg.training.get('output_dir', 'checkpoints'))
    os.makedirs(checkpath, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(checkpath, 'config.yaml'))

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor, mode='min' if metric == 'mse' else 'max',
        dirpath=checkpath, filename=filename, save_last=True,
    )
    
    logger = False
    if cfg.project is not None:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=cfg.project, name=cfg.name, log_model=False)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=100,
        max_epochs=cfg.training.epochs,
        accelerator=cfg.platform.get('accel', 'auto'),
        devices=1,
        limit_train_batches=cfg.training.batch_fraction,
        fast_dev_run=cfg.training.get('fast_dev_run', False),
    )
    
    trainer.fit(model_pl, train_loader, val_loader, ckpt_path=cfg.training.get('ckpt'))
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_config", help="YAML with dataset/checkpoint paths")
    parser.add_argument("training_config", help="YAML with model and training options")
    parser.add_argument(
        "--fast-dev-run", action="store_true",
        help="Run one training and validation batch without saving checkpoints",
    )
    parser.add_argument(
        "--resume", help="Resume optimizer, scheduler, and model from a Lightning checkpoint",
    )
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.local_config), OmegaConf.load(args.training_config))
    if args.fast_dev_run:
        OmegaConf.update(cfg, 'training.fast_dev_run', True)
    if args.resume:
        OmegaConf.update(cfg, 'training.ckpt', args.resume)
    train(cfg)


if __name__ == "__main__":
    main()
