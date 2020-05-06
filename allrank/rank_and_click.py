import numpy as np
import os
import torch
from allrank.click_models.base import RandomClickModel
from allrank.click_models.click_utils import click_on_listings
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.inference.inference_utils import rank_listings
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.utils.command_executor import execute_command
from allrank.utils.file_utils import create_output_dirs, PathsContainer
from allrank.utils.ltr_logging import init_logger
from argparse import ArgumentParser, Namespace
from attr import asdict
from pprint import pformat


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank rank and apply click model")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")

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

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Will use device {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))

    # load_model_dict

    model.to(dev)

    train_listings = rank_listings(train_dl, model, dev)
    val_listings = rank_listings(val_dl, model, dev)

    click_model = RandomClickModel(2)

    train_click_listings = click_on_listings(train_listings, click_model, False)
    val_click_listings = click_on_listings(val_listings, click_model, False)

    # save clickthrough dataset
    return train_click_listings, val_click_listings


if __name__ == "__main__":
    run()
