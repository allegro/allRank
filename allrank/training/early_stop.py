from allrank.utils.ltr_logging import get_logger

logger = get_logger()


class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = 0.0
        self.best_epoch = 0

    def step(self, current_value, current_epoch):
        logger.info("Current:{} Best:{}".format(current_value, self.best_value))
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = current_epoch

    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience
