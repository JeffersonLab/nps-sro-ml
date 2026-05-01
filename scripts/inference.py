#!/usr/bin/env python3

import collections
import argparse
import pathlib
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pytorch_src"))

from inference.oc_inference import (
    OcInferenceHyperparameters,
    build_oc_inferencer,
)
from utils.utils import prepare_device, import_attr
from utils.config import ConfigParser


def main(cfg: ConfigParser):

    logger = cfg.get_logger('inference')
    dl = cfg.init_obj("data_loader", logger=logger)
    vdl = dl.split_validation()

    model_path = pathlib.Path(cfg.get("model_pth"))
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    device, _ = prepare_device(cfg.get("n_gpu"))
    logger.info("Using device: {}".format(device))

    model = model.to(device)
    model.eval()

    hyperparams = OcInferenceHyperparameters.from_mapping(cfg.get("hyperparameters", {}))
    inferencer = build_oc_inferencer(
        model,
        config=cfg.config,
        hyperparameters=hyperparams,
    )
    results = inferencer.infer_dataloader(tqdm(vdl))

    df = results.to_dataframe()

    out_dir = cfg.get("save_dir") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "results.csv", index=False)
    logger.info(f"Saved inference results to {out_dir / 'results.csv'}")


def load_model(resume):
    checkpoint = torch.load(str(resume.absolute()), map_location="cpu")
    state_dict = checkpoint['state_dict']
    model, _ = import_attr(
        checkpoint['arch'],
        avail_modules=[checkpoint['module']],
        **checkpoint['metadata'],
    )

    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inference script for OC model on NPS data"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        default="config/inference/geant4.json",
        help="Path to a config file.",
    )
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(
            ['-m', '--model'],
            type=pathlib.Path,
            target='model_pth',
        ),
    ]
    cfg = ConfigParser.from_args(parser, options)
    main(cfg)
