"""CLI entry point for evaluating a trained ThermoMPNN-D checkpoint.

Paper map: selects the standard or Siamese Lightning wrapper, loads its
checkpoint, and delegates batched prediction to ``v2_inference``.
"""

import torch
import argparse
from omegaconf import OmegaConf
import pandas as pd
import os

from mutation_prediction.thermo_mpnn_d.thermompnn.inference.v2_inference import load_v2_dataset, run_prediction_batched
from mutation_prediction.thermo_mpnn_d.thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from mutation_prediction.thermo_mpnn_d.thermompnn.train_thermompnn import parse_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file for model and dataset parameters', type=str, default='../config.yaml')
    parser.add_argument('--model', help='path to the trained model checkpoint', type=str, default='checkpoint.ckpt')
    parser.add_argument('--local', help='config file for data and pretrained-weight paths', type=str, default='../local.yaml')
    parser.add_argument('--keep_preds', action='store_true', default=False,
                        help='Save per-mutation predictions to CSV')
    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.load(args.local))
    cfg = parse_cfg(cfg)

    with torch.no_grad():
        inference(cfg, args)


def inference(cfg, args):
    """Catch-all inference function for all ThermoMPNN versions"""

    # pre-initialization params
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ds_name = cfg.data.dataset
    model_name = args.model.removesuffix('.ckpt')
    
    ds = load_v2_dataset(cfg)
    print('Loading model %s' % args.model)
    if cfg.model.aggregation == 'siamese':
        model_class = TransferModelPLv2Siamese
    else:
        model_class = TransferModelPLv2
    model = model_class.load_from_checkpoint(
        args.model, cfg=cfg, map_location=device
    ).model
    results = run_prediction_batched(model_name, model, ds_name, ds, [], args.keep_preds, cfg=cfg)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"ThermoMPNN_{os.path.basename(model_name).removesuffix('.ckpt')}_{ds_name}_metrics.csv")

    return df


if __name__ == "__main__":
    main()