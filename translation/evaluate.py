import os.path
import pickle

import hydra
from typing import Union

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import load_data, split_data, get_preprocessing, get_dataloaders
from src.params.main_params import get_main_params, MainParams
from src.train import evaluate_model


config_path = "config_archive/experiment_2"


@hydra.main(config_path=config_path, config_name="config.yaml")
def main(cfg: Union[DictConfig, MainParams]) -> None:
    if isinstance(cfg, MainParams):
        main_params = cfg
    else:
        main_params = get_main_params(dict(cfg))

    preprocessing_path = "/home/juliarty/Documents/WorkArea/MADE/nlp/translation/outputs/seq2seq_bidirectional/preprocessing.pkl"
    model_path = "/home/juliarty/Documents/WorkArea/MADE/nlp/translation/outputs/seq2seq_bidirectional/outputs/seq2seq_bidirectional/tmp_model.pt"

    with open(preprocessing_path, 'rb') as f:
        preprocessing = pickle.load(f)
    _, _, test_dl = get_dataloaders(
        main_params.dataset, preprocessing, main_params.train.batch_size
    )

    writer = SummaryWriter(os.path.join(config_path, "tensorboard_logdir"))
    evaluate_model(
        model_path=model_path,
        preprocessing=preprocessing,
        dataset_params=main_params.dataset,
        test_dl=test_dl,
        device=main_params.train.device,
        summary_writer=writer,
    )


if __name__ == "__main__":
    main()
