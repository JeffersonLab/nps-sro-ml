#!/usr/bin/env python3

import collections
import argparse
import pathlib
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pytorch_src"))

from utils.utils import prepare_device, import_attr
from models.oc_inference import (
    OcInferenceHyperparameters,
    OcInferenceResults,
    oc_inference_per_graph,
)
from utils.graph import create_unique_object_ids
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

    results = OcInferenceResults()
    hyperparams = OcInferenceHyperparameters.from_mapping(cfg.get("hyperparameters", {}))

    event_counter = 0
    with torch.no_grad():
        for data in tqdm(vdl):
            data = data.to(device)
            x = data.x
            y = data.y.squeeze(-1).long()
            pos = data.pos
            batch = getattr(data, "batch", None)

            object_ids = create_unique_object_ids(y, batch, noise_idx=hyperparams.noise_idx)
            x_c, beta = model(x, pos, batch=batch)
            beta = beta.squeeze(-1)

            if batch is None:
                batch = torch.zeros(x_c.size(0), dtype=torch.long, device=device)

            for b in batch.unique(sorted=True):
                b_mask = batch == b

                x_c_b = x_c[b_mask]
                beta_b = beta[b_mask]
                pos_b = pos[b_mask]

                cluster_ids, min_d = oc_inference_per_graph(
                    x_c_b,
                    beta_b,
                    beta_thres=hyperparams.beta_thres,
                    dist_thres=hyperparams.dist_thres,
                    bkg_idx=hyperparams.noise_idx,
                )

                results.append_graph(
                    event_id=event_counter,
                    cluster_ids=cluster_ids,
                    min_d=min_d,
                    beta=beta_b,
                    x_c=x_c_b,
                    object_ids=object_ids[b_mask],
                    pos=pos_b,
                )
                event_counter += 1

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
