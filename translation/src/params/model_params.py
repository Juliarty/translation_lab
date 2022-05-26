import dataclasses


@dataclasses.dataclass()
class ModelParams:
    enc_emb_dim: int
    dec_emb_dim: int
    enc_hid_dim: int
    dec_hid_dim: int
    attn_dim: int
    enc_dropout: float
    dec_dropout: float
    pretrained_embedding: str
    rnn_type: str
    bidirectional: bool
