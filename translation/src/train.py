import math
import random
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import tqdm

from nltk.translate.bleu_score import corpus_bleu
from torch.utils.tensorboard import SummaryWriter

from src.data.preprocessing import Preprocessing
from src.params.dataset_params import DatasetParams

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
    epoch: int,
    summary_writer: SummaryWriter,
    debug: bool,
):

    model.train()

    epoch_loss = 0

    for i, (src, trg) in enumerate(train_dl):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        summary_writer.add_scalar("train_loss", loss.item(), epoch * len(train_dl) + i)

        if debug and i > 2:
            break

    return epoch_loss / len(train_dl)


def evaluate(
    model: nn.Module,
    val_dl: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    summary_writer: SummaryWriter,
    debug: bool,
):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, (src, trg) in enumerate(val_dl):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            summary_writer.add_scalar("val_loss", loss.item(), epoch * len(val_dl) + i)

            if debug and i > 2:
                break

    return epoch_loss / len(val_dl)


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    valid_iter: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    n_epochs: int,
    clip: float,
    device: torch.device,
    summary_writer: SummaryWriter,
    debug: bool,
    tmp_model_save_path: str,
) -> nn.Module:
    best_val_loss = float("inf")
    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_dl,
            optimizer,
            criterion,
            clip,
            device,
            epoch,
            summary_writer,
            debug,
        )

        valid_loss = evaluate(
            model, valid_iter, criterion, device, epoch, summary_writer, debug
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            with open(tmp_model_save_path, "wb") as f:
                torch.save(model, f)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        logger.info(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        logger.info(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )
        if debug and epoch > 2:
            break

    return model


def evaluate_blue(
    model: nn.Module,
    preprocessing: Preprocessing,
    test_iterator: torch.utils.data.DataLoader,
    dataset_params: DatasetParams,
    device: torch.device,
    summary_writer: SummaryWriter,
    debug: bool,
    log_translation_count: int = 100
) -> float:
    original_text = []
    generated_text = []
    source_text = []
    model.eval()
    with torch.no_grad():
        for i, (src, trg) in tqdm.tqdm(
            enumerate(test_iterator), total=len(test_iterator)
        ):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output.argmax(dim=-1)

            source_text.extend(
                [
                    preprocessing.itos(t, language=dataset_params.source_language)
                    for t in src.detach().cpu().T
                ]
            )

            original_text.extend(
                [
                    preprocessing.itos(t, language=dataset_params.target_language)
                    for t in trg.detach().cpu().T
                ]
            )

            generated_text.extend(
                [
                    preprocessing.itos(t, language=dataset_params.target_language)
                    for t in output.detach().cpu().T
                ]
            )

            if debug and i > 2:
                break

    for i in range(log_translation_count):
        origin = " ".join(
            [
                word
                for word in source_text[i]
                if word not in preprocessing.SPECIAL_TOKENS
            ]
        )
        translated = " ".join(
            [
                word
                for word in original_text[i]
                if word not in preprocessing.SPECIAL_TOKENS
            ]
        )

        end_index = len(generated_text)

        if preprocessing.END_TOKEN in generated_text:
            end_index = generated_text.index(preprocessing.END_TOKEN)

        generated = " ".join(
            [
                word
                for word in generated_text[:end_index][i]
                if word not in preprocessing.SPECIAL_TOKENS
                or word == preprocessing.UNKNOWN_TOKEN
            ]
        )
        summary_writer.add_text(
            "Translation",
            f"origin: {origin}\n\ntranslated: {translated}\n\ngenerated: {generated}\n\n",
            i,
        )

    return corpus_bleu([[text] for text in original_text], generated_text) * 100


def evaluate_model(
    model_path: str,
    preprocessing: Preprocessing,
    test_dl: torch.utils.data.DataLoader,
    dataset_params: DatasetParams,
    device: torch.device,
    summary_writer: SummaryWriter,
):
    with open(model_path, "rb") as f:
        model = torch.load(f)
        logger.info(f"Loaded model {model}")

        score = evaluate_blue(
            model,
            preprocessing=preprocessing,
            test_iterator=test_dl,
            dataset_params=dataset_params,
            device=device,
            summary_writer=summary_writer,
            debug=False,
        )

        logger.info(f"BLEU score: {score}")
