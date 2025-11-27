import pathlib
import json
from typing import Optional
from src.config import JSON_LOG_DIR

class JSONLogger:
    """Logger for storing training metrics and configuration in JSON format (Alternative to Weights & Biases).
    Args:
        model_name (str): Name of the model used for logging.
        config (dict): Configuration dictionary containing model and training parameters.
        save_frequency (int, optional): Log file will be saved auf `save_frequency` logging steps. Defaults to 10.
        impute_time (Optional[float], optional): Time taken for imputation, if applicable. Defaults to None.
    """

    def __init__(self, model_name: str, config: dict, save_frequency: int = 10, impute_time: Optional[float] = None) -> None:
        self.model_name = model_name
        self.config = config
        self.log_file_path = JSON_LOG_DIR / pathlib.Path(model_name + '.json')
        self.log_dict = None
        self.global_step = 0
        self.save_frequency = save_frequency
        self.impute_time = impute_time

    def __enter__(self):
        self.log_dict = {}
        self.log_dict['name'] = self.model_name
        self.log_dict['config'] = self.config
        self.log_dict['logs'] = {}
        if self.impute_time is not None:
            self.log_dict['impute_time'] = self.impute_time
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        return None

    def log(self, logs: dict):
        assert self.log_dict is not None, "Use logger as context manager"
        self.log_dict['logs'][self.global_step] = logs
        self.global_step += 1
        if self.global_step % self.save_frequency == 0:
            self.save()

    def save(self):
        with open(self.log_file_path, 'w') as file:
            json.dump(self.log_dict, file, indent=2)
    
class DummyLogger:
    """Dummy logger used for debugging.
    """
    def __init__(self):
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def log(self, logs: dict):
        pass

    