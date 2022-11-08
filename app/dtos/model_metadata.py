from dataclasses import dataclass

from app.dtos.train import (TrainMetadata, TrainMetadataFields, TrainResult,
                            TrainResultFields)


@dataclass(frozen=True)
class ModelMetadata(TrainMetadata, TrainResult):
    pass


@dataclass(frozen=True)
class ModelMetadataFields(TrainMetadataFields, TrainResultFields):
    pass
