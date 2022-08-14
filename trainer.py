import argparse
import logging
import os
import time

from datasets.dataloader import create_dataloader
from utils.hparams import HParam
from utils.train import train
from utils.writer import MyWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output of the model for logging, saving checkpoint",
    )
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(output_dir, "%d.log" % (time.time()))
            ),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, output_dir)

    assert hp.audio.hop_length == 256, (
        "hp.audio.hop_length must be equal to 256, got %d" % hp.audio.hop_length
    )
    assert hp.data.train != "" and hp.data.validation != "", (
        "hp.data.train and hp.data.validation can't be empty: please fix %s"
        % args.config
    )

    trainloader = create_dataloader(hp, args, True)
    valloader = create_dataloader(hp, args, False)

    train(
        args,
        output_dir,
        trainloader,
        valloader,
        writer,
        logger,
        hp,
        hp_str,
    )
