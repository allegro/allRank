import os
from argparse import ArgumentParser, Namespace
from pprint import pformat

import numpy as np
import torch
from attr import asdict

from allrank.click_models.base import RandomClickModel
from allrank.click_models.click_utils import click_on_listings
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset
from allrank.data.dataset_saving import write_to_libsvm_without_masked
from allrank.inference.inference_utils import rank_listings
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel, load_state_dict_from_file
from allrank.utils.command_executor import execute_command
from allrank.utils.config_utils import instantiate_from_recursive_name_args
from allrank.utils.file_utils import create_output_dirs, PathsContainer
from allrank.utils.ltr_logging import init_logger


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank rank and apply click model")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with model config")
    parser.add_argument("--input-model-path", required=True, type=str, help="Path to the model to read weights")

    return parser.parse_args()


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    os.makedirs(paths.base_output_path, exist_ok=True)

    create_output_dirs(paths.output_dir)
    logger = init_logger(paths.output_dir)

    logger.info("will save data in {output_dir}".format(output_dir=paths.base_output_path))

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # LibSVMDatasets
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # gpu support
    dev = get_torch_device()
    logger.info("Will use device {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))

    model.load_state_dict(load_state_dict_from_file(args.input_model_path, dev))
    logger.info(f"loaded model weights from {args.input_model_path}")

    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    train_listings, val_listings = rank_listings(train_ds, val_ds, model, config)

    assert config.click_model is not None, "click_model must be defined in config for this run"
    click_model = instantiate_from_recursive_name_args(name_args=config.click_model)

    train_click_listings = click_on_listings(train_listings, click_model, False)
    val_click_listings = click_on_listings(val_listings, click_model, False)

    # save clickthrough dataset
    write_to_libsvm_without_masked(os.path.join(paths.output_dir, "train.txt"), *train_click_listings)
    write_to_libsvm_without_masked(os.path.join(paths.output_dir, "valid.txt"), *val_click_listings)


if __name__ == "__main__":
    run()
