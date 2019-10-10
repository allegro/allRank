from typing import Any, Dict

import tensorflow as tf


class TensorboardSummaryWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.writers = {}  # type: Dict[str, Any]

    def ensure_writer_exists(self, name):
        if name not in self.writers.keys():
            self.writers[name] = tf.summary.FileWriter(
                "{path}/{name}".format(path=self.output_path, name=name))

    def save_to_tensorboard(self, results, n_epoch):
        for (role, metric), value in results.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=metric, simple_value=value)])
            metric_with_role = "_".join([metric, role])
            self.ensure_writer_exists(metric_with_role)
            self.writers[metric_with_role].add_summary(summary, n_epoch)
            self.writers[metric_with_role].flush()
