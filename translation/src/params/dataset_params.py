import dataclasses


@dataclasses.dataclass()
class DatasetParams:
    dataset_path: str
    dataset_url: str
    source_language: str
    target_language: str
    source_language_index: int
    target_language_index: int
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    train_size: float
    val_size: float
    test_size: float
