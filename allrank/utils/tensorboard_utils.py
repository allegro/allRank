import os
from typing import Any, Dict, Tuple

from tensorboardX import SummaryWriter


class TensorboardSummaryWriter:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.writers = {}  # type: Dict[str, Any]

    def ensure_writer_exists(self, name: str) -> None:
        if name not in self.writers.keys():
            writer_path = os.path.join(self.output_path, name)
            self.writers[name] = SummaryWriter(writer_path)

    def save_to_tensorboard(self, results: Dict[Tuple[str, str], float], n_epoch: int) -> None:
        for (role, metric), value in results.items():
            metric_with_role = "_".join([metric, role])
            self.ensure_writer_exists(metric_with_role)
            self.writers[metric_with_role].add_scalar(metric, value, n_epoch)

    def close_all_writers(self) -> None:
        for writer in self.writers.values():
            writer.close()
