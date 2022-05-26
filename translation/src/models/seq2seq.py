import random
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else:
            self.embedding = pretrained_embedding

        if rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=bidirectional, num_layers=self.num_layers, device=device)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=bidirectional, num_layers=self.num_layers, device=device)

        if bidirectional:
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim, device=device)
        else:
            self.fc = nn.Linear(enc_hid_dim, dec_hid_dim, device=device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Any, Tuple[Any, Any]]:

        embedded = self.dropout(self.embedding(src))

        if self.rnn_type == 'gru':
            outputs, hidden = self.rnn(embedded)
        elif self.rnn_type == 'lstm':
            outputs, (hidden, context) = self.rnn(embedded)
        else:
            raise Exception()

        if self.bidirectional:
            hidden = torch.tanh(
                self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            )
        else:
            hidden = torch.tanh(self.fc(hidden))

        if self.rnn_type == 'gru':
            return outputs, hidden.squeeze(0)
        elif self.rnn_type == 'lstm':
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

        self.attn = nn.Linear(self.attn_in, attn_dim, device=device)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2))
        )

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


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
        pretrained_embedding: nn.Module=None
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
            self.embedding = nn.Embedding(output_dim, emb_dim, device=device)
        else:
            self.embedding = pretrained_embedding

        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.enc_hid_dim + self.emb_dim , self.dec_hid_dim, device=device)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.enc_hid_dim + self.emb_dim , self.dec_hid_dim, device=device)

        self.out = nn.Linear(
            self.attention.attn_in + self.emb_dim, self.output_dim, device=device
        )

        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(
        self, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(
        self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tuple[Any, Tuple[Any, Any]]:

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        if self.rnn_type == 'gru':
            weighted_encoder_rep = self._weighted_encoder_rep(
                decoder_hidden, encoder_outputs
            )

            rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
            output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        elif self.rnn_type == 'lstm':
            if not isinstance(decoder_hidden, Tuple):
                decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))

            weighted_encoder_rep = self._weighted_encoder_rep(
                decoder_hidden[0], encoder_outputs
            )

            rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
            output, decoder_hidden = self.rnn(rnn_input, (decoder_hidden[0].unsqueeze(0), decoder_hidden[1].unsqueeze(0)))
        else:
            raise Exception()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        if self.rnn_type == 'gru':
            return output, decoder_hidden.squeeze(0)
        elif self.rnn_type == 'lstm':
            return output, (decoder_hidden[0].squeeze(0), decoder_hidden[1].squeeze(0))


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

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
