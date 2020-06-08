import os
import tempfile
from typing import Any
from urllib.parse import urlparse

import gcsfs
from attr import attrib, attrs
from pkg_resources import Requirement, resource_filename

from allrank.utils.command_executor import execute_command
from allrank.utils.ltr_logging import get_logger

logger = get_logger()


@attrs
class PathsContainer:
    local_base_output_path = attrib(type=str)
    base_output_path = attrib(type=str)
    output_dir = attrib(type=str)
    tensorboard_output_path = attrib(type=str)
    config_path = attrib(type=str)

    @classmethod
    def from_args(cls, output, run_id, config_path, package_name="allrank"):
        base_output_path = get_path_from_local_uri(output)
        if is_gs_path(base_output_path):
            local_base_output_path = tempfile.mkdtemp()
        else:
            local_base_output_path = base_output_path
        output_dir = os.path.join(local_base_output_path, "results", run_id)
        tensorboard_output_path = os.path.join(base_output_path, "tb_evals", "single", run_id)
        if not os.path.exists(config_path):
            print("config not exists at {}, extracting config file path from package {}".format(config_path, package_name))
            config_path = resource_filename(Requirement.parse(
                package_name), os.path.join(package_name, config_path))
        print("will read config from {}".format(config_path))
        return cls(local_base_output_path, base_output_path, output_dir, tensorboard_output_path, config_path)


def clean_up(path):
    rm_command = "rm -rf {path}".format(path=path)
    execute_command(rm_command)


def create_output_dirs(output_path: str) -> None:
    for subdir in ["models", "models/partial", "evals", "evals/tensorboard", "predictions"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)


def get_path_from_local_uri(uri: Any) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return parsed.netloc + parsed.path
    else:
        return uri


def is_gs_path(uri) -> bool:
    return urlparse(uri).scheme == "gs"


def open_local_or_gs(path, mode):
    open_func = gcsfs.GCSFileSystem().open if is_gs_path(path) else open
    return open_func(path, mode)


def copy_file_to_local(uri: str) -> str:
    temp_dir = tempfile.mkdtemp()
    local_file = "local_file"
    command = "gsutil cp {gs_uri} {local_path}".format(
        gs_uri=uri, local_path=os.path.join(temp_dir, local_file))
    execute_command(command)
    return os.path.join(temp_dir, local_file)


def copy_local_to_gs(source_local: str, destination_uri: str) -> None:
    command = "gsutil cp -r {source_local}/* {destination_uri}".format(
        source_local=source_local, destination_uri=destination_uri)
    execute_command(command)
