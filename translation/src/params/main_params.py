import dataclasses
import logging

import torch
from marshmallow_dataclass import class_schema

from src.params.dataset_params import DatasetParams
from src.params.model_params import ModelParams
from src.params.preprocessing_params import PreprocessingParams
from src.params.train_params import TrainParams

logger = logging.getLogger(__name__)


@dataclasses.dataclass()
class MainParams:
    dataset: DatasetParams
    model: ModelParams
    train: TrainParams
    preprocessing: PreprocessingParams
    output_path: str
    log_dir: str
    tensorboard_logdir: str
    debug: bool


MainParamsSchema = class_schema(MainParams)


def get_main_params(dict_config: dict) -> MainParams:
    logger.info(f"Load config: {dict_config}")
    return MainParamsSchema().load(dict_config)
