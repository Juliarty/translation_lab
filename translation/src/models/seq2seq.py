import random
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, BertModel


class Encoder(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        bidirectional: bool,
        input_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
        device: torch.device,
        pretrained_embedding: nn.Module = None,
        is_embedded: bool = True,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim

        self.dec_hid_dim = dec_hid_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_layers = 1
        self.is_embedded = is_embedded

        if not is_embedded:
            if pretrained_embedding is None:
                self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
            else:
                self.embedding = pretrained_embedding

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                emb_dim,
                enc_hid_dim,
                bidirectional=bidirectional,
                num_layers=self.num_layers,
                device=device,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                emb_dim,
                enc_hid_dim,
                bidirectional=bidirectional,
                num_layers=self.num_layers,
                device=device,
            )

        if bidirectional:
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim, device=device)
        else:
            self.fc = nn.Linear(enc_hid_dim, dec_hid_dim, device=device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Any, Tuple[Any, Any]]:
        if self.is_embedded:
            embedded = self.dropout(src)
        else:
            embedded = self.embedding(src)

        if self.rnn_type == "gru":
            outputs, hidden = self.rnn(embedded)
        elif self.rnn_type == "lstm":
            outputs, (hidden, context) = self.rnn(embedded)
        else:
            raise Exception()

        if self.bidirectional:
            hidden = torch.tanh(
                self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            )
        else:
            hidden = torch.tanh(self.fc(hidden))

        if self.rnn_type == "gru":
            return outputs, hidden.squeeze(0)
        elif self.rnn_type == "lstm":
            return outputs, hidden.squeeze(0)


class Attention(nn.Module):
    def __init__(
        self,
        enc_hid_dim: int,
        dec_hid_dim: int,
        attn_dim: int,
        device: torch.device,
        encoder_bidirectional: bool,
    ):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        if encoder_bidirectional:
            self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        else:
            self.attn_in = enc_hid_dim + dec_hid_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.attn_in, attn_dim, bias=False, device=device),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, device=device),
        )

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, src_len, hidden)

        # Bahdanau attention
        # (batch, src_len, score)
        attention = self.mlp(
            torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)
        )
        attention = F.softmax(attention, dim=1)

        attention = attention.permute(0, 2, 1)  # (batch, score, src_len)
        # sum all encoder_outputs weighted with attention
        return torch.bmm(attention, encoder_outputs).squeeze(1)  # batch, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        bidirectional_encoder: bool,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
        attention: nn.Module,
        device: torch.device,
        pretrained_embedding: nn.Module = None,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        if bidirectional_encoder:
            self.enc_hid_dim *= 2

        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        if pretrained_embedding is None:
            self.embedding = nn.Embedding(output_dim, self.emb_dim, device=device)
        else:
            self.embedding = pretrained_embedding

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                self.enc_hid_dim + self.emb_dim, self.dec_hid_dim, device=device
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                self.enc_hid_dim + self.emb_dim, self.dec_hid_dim, device=device
            )

        self.out = nn.Linear(
            self.attention.attn_in + self.emb_dim, self.output_dim, device=device
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tuple[Any, Tuple[Any, Any]]:

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        if self.rnn_type == "gru":
            encoder_context = self.attention(decoder_hidden, encoder_outputs)

            rnn_input = torch.cat((embedded, encoder_context), dim=2)
            output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        elif self.rnn_type == "lstm":
            if not isinstance(decoder_hidden, Tuple):
                decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))

            encoder_context = self.attention(
                decoder_hidden[0], encoder_outputs
            ).unsqueeze(0)

            rnn_input = torch.cat((embedded, encoder_context), dim=2)
            output, decoder_hidden = self.rnn(
                rnn_input,
                (decoder_hidden[0].unsqueeze(0), decoder_hidden[1].unsqueeze(0)),
            )
        else:
            raise Exception()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        encoder_context = encoder_context.squeeze(0)

        output = self.out(torch.cat((output, encoder_context, embedded), dim=1))

        if self.rnn_type == "gru":
            return output, decoder_hidden.squeeze(0)
        elif self.rnn_type == "lstm":
            return output, (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.rubert_emb = AutoModel.from_pretrained(
            "../../deeppavlov/rubert_cased_L-12_H-768_A-12_pt"
        )
        for param in self.rubert_emb.parameters():
            param.requires_grad = False

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.linear = nn.Linear(768, self.encoder.emb_dim)

    def forward(
        self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        src = self.linear(self.rubert_emb(src).last_hidden_state)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = trg[t] if teacher_force else top1

        return outputs
