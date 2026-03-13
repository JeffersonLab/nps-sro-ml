#!/usr/bin/env python3

import collections
import torch
import argparse

from utils.utils import prepare_device
from utils.config import ConfigParser


def main(cfg: ConfigParser):
    """
    Run the training pipeline. The function initializes the dataset, model, optimizer, and trainer based on the provided configuration, and then starts the training process. If the `debug` flag is set in the configuration, it will run a single forward pass with one batch of data.
    """
    logger = cfg.get_logger('train')
    dl = cfg.init_obj("data_loader", logger=logger)
    vdl = dl.split_validation()

    model = cfg.init_obj("arch")
    logger.info(model)

    device, device_ids = prepare_device(cfg.get("n_gpu"))
    logger.info("Using device: {}".format(device))
    model = model.to(device)
    if len(device_ids) > 1:
        logger.info('Multi-GPU mode: using {}'.format(device_ids))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = cfg.init_obj("optimizer", trainable_params)
    lr_scheduler = cfg.init_obj("lr_scheduler", optimizer)

    trainer_cls = cfg.init_obj("trainer")
    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        device=device,
        dataloader=dl,
        valid_dataloader=vdl,
        lr_scheduler=lr_scheduler,
        logger=logger,
        config=cfg["trainer"],
    )

    if cfg.get("debug"):
        logger.info(
            "Running in debug mode with a single batch. Test forward pass by exporting model to ONNX."
        )
        trainer.export_onnx("my_model.onnx")
        return

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for NPS SRO ML project."
    )

    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to a config file."
    )

    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with a single batch."
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(
            ['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'
        ),
    ]
    cfg = ConfigParser.from_args(parser, options)
    main(cfg)
