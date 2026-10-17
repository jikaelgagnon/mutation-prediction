print("Starting imports", flush=True)

import os
print("Imported os", flush=True)

import argparse
print("Imported argparse", flush=True)

from torch.utils.data import DataLoader
print("Imported DataLoader", flush=True)

import pytorch_lightning as pl
print("Imported pytorch_lightning", flush=True)

from pytorch_lightning.callbacks import ModelCheckpoint
print("Imported ModelCheckpoint", flush=True)

from omegaconf import OmegaConf
print("Imported OmegaConf", flush=True)

from mutation_prediction.thermo_mpnn_d.thermompnn.parsers import get_v2_dataset
print("Imported get_v2_dataset", flush=True)

from mutation_prediction.model.model import PooledAttentionModel, PooledAttentionModelPL
print("Imported PooledAttentionModel and PooledAttentionModelPL", flush=True)

from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import tied_featurize_mut
print("Imported tied_featurize_mut", flush=True)

print("All imports complete", flush=True)

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

    return cfg


def train(cfg):
    print('Configuration:\n', cfg)

    cfg = parse_cfg(cfg)

    if cfg.project is not None:
        import wandb

        wandb.init(project=cfg.project, name=cfg.name)

    train_dataset, val_dataset = get_v2_dataset(cfg)

    train_loader = DataLoader(train_dataset, 
                                collate_fn=lambda b: tied_featurize_mut(b, side_chains=cfg.data.side_chains), 
                                shuffle=cfg.training.shuffle, 
                                num_workers=cfg.training.num_workers, 
                                batch_size=cfg.training.batch_size)
    val_loader = DataLoader(val_dataset, 
                                collate_fn=lambda b: tied_featurize_mut(b, side_chains=cfg.data.side_chains), 
                                shuffle=False, 
                                num_workers=cfg.training.num_workers, 
                                batch_size=cfg.training.batch_size)

    model_pl = PooledAttentionModelPL(cfg)
    
    # additional params, logging, checkpoints for training
    filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.4f}'
    monitor = 'val_ddG_spearman'
    
    checkpath = os.path.abspath(cfg.training.get('output_dir', 'checkpoints'))
    os.makedirs(checkpath, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath=checkpath, filename=filename)
    
    logger = None
    if cfg.project is not None:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=cfg.project, name=cfg.name, log_model=False)
    n_steps = 100
    
    trainer = pl.Trainer(callbacks=[checkpoint_callback], 
                        logger=logger, 
                        log_every_n_steps=n_steps, 
                        max_epochs=cfg.training.epochs,
                        accelerator=cfg.platform.accel, 
                        devices=1, 
                        limit_train_batches=cfg.training.batch_fraction, 
    )
    
    trainer.fit(model_pl, train_loader, val_loader) #, ckpt_path=cfg.training.ckpt)


def main():
    print("Starting main function")
    parser = argparse.ArgumentParser(description="Train our new model.")
    parser.add_argument("local_config", help="YAML with dataset/checkpoint paths")
    parser.add_argument("training_config", help="YAML with model and training options")
    args = parser.parse_args()
    print("Loading config")
    cfg = OmegaConf.merge(OmegaConf.load(args.local_config), OmegaConf.load(args.training_config))
    print("Starting training")
    train(cfg)

if __name__ == "__main__":
    main()
