import dataclasses


@dataclasses.dataclass()
class PreprocessingParams:
    spacy_tokenizer: dict
