import dataclasses


@dataclasses.dataclass()
class TrainParams:
    batch_size: int
    n_epoch: int
    device: str
    learning_rate: float
