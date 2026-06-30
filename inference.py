import os
import time
import argparse

from dataset import build_dataloader
from networks import build_model 
from processor import do_inference
from utils.logger import setup_logger
from utils.metrics import set_seed
from utils.iotools import load_train_configs
from utils.checkpointer import Checkpointer

import torch
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    # parser.add_argument('--config_file', default='output\MNIST/20260626_232650_baseline\configs.yaml')
    parser.add_argument('--config_file', default='output\MNIST/20260630_113542_edge\configs.yaml')
    parser.add_argument("--brightness", type=float, default=1.0,    choices=[1.0, 0.6, 0.2])
    cli_args = parser.parse_args()
    cfg_args = load_train_configs(cli_args.config_file)
    cfg_args.training = False
    cfg_args.use_edge = False
    cfg_args.brightness = cli_args.brightness

    set_seed(cfg_args.seed)
    args = cfg_args
    logger = setup_logger('Edge.test', save_dir=args.output_dir, if_train=args.training)
    # logger.info(args)
    device = 'cuda'

    test_loader, num_classes = build_dataloader(args, logger)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=os.path.join(args.output_dir, 'best.pth'), )
    model.to(device)
    do_inference(model, test_loader)

    