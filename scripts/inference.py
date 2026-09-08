#!/usr/bin/env python3

import collections
import argparse
import pathlib
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pytorch_src"))

from inference.oc_inference import (
    BaseOcInferenceHyperparameters,
    BaseOcInferenceResults,
)
from utils.config import ConfigParser
from utils.utils import import_attr, prepare_device


def main(cfg: ConfigParser):
    logger = cfg.get_logger("inference")
    dl = cfg.init_obj("data_loader", logger=logger)
    vdl = dl.split_validation()

    model_path = pathlib.Path(cfg.get("model_pth"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    device, _ = prepare_device(cfg.get("n_gpu"))
    logger.info(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    inference_cls = cfg.init_obj("inference")
    inferencer = inference_cls(
        model,
        hyperparameters=cfg["inference"],
    )
    results: BaseOcInferenceResults = inferencer.infer(tqdm(vdl))

    save_dir = pathlib.Path(cfg.get("save_dir", None))
    if save_dir is None or not save_dir.exists():
        save_dir = pathlib.Path(model_path.parent)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / "results.csv"
    results.to_csv(results_path, index=False)
    logger.info(f"Saved inference results to {results_path}")


def load_model(resume):
    checkpoint = torch.load(str(resume.absolute()), map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model, _ = import_attr(
        checkpoint["arch"],
        avail_modules=[checkpoint["module"]],
        **checkpoint["metadata"],
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
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["-m", "--model"],
            type=pathlib.Path,
            target="model_pth",
        ),
    ]
    cfg = ConfigParser.from_args(parser, options)
    main(cfg)
