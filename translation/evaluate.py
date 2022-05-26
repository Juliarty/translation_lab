import os.path
import pickle
import time
import hydra
from typing import Union

import torch
import tqdm
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import load_data, split_data, get_preprocessing, get_dataloaders
from src.data.preprocessing import Preprocessing
from src.params.main_params import get_main_params, MainParams
from src.train import evaluate_model


config_path = "config_archive/experiment_1"


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


@hydra.main(config_path=config_path, config_name="config.yaml")
def measure_inference_speed(cfg: Union[DictConfig, MainParams]) -> float:
    batch_size = 32
    if isinstance(cfg, MainParams):
        main_params = cfg
    else:
        main_params = get_main_params(dict(cfg))

    preprocessing_path = "/home/juliarty/Documents/WorkArea/MADE/nlp/translation/outputs/seq2seq_2/preprocessing.pkl"
    model_path = "/home/juliarty/Documents/WorkArea/MADE/nlp/translation/outputs/seq2seq_1/model.pt"

    with open(model_path, "rb") as f:
        model = torch.load(f)

    with open(preprocessing_path, 'rb') as f:
        preprocessing = pickle.load(f)
    _, _, test_dl = get_dataloaders(
        main_params.dataset, preprocessing, batch_size
    )
    device = 'cuda'
    generated_text = []
    model.eval()

    start_time = time.time()

    batches_num = len(test_dl)
    with torch.no_grad():
        for i, (src, trg) in enumerate(test_dl):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)
            output = output.argmax(dim=-1)
            generated_text.extend(
                [
                    preprocessing.itos(t, language=main_params.dataset.target_language)
                    for t in output.detach().cpu().T
                ]
            )

    end_time = time.time()

    print((end_time - start_time) / batches_num)


if __name__ == "__main__":
    measure_inference_speed()
