from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelInfo:
    api_name: str
    storage_dir: str

    @classmethod
    def with_repo_suffix(cls, api_name: str, repo_suffix: str | None = None) -> "ModelInfo":
        suffix = repo_suffix or api_name
        return cls(api_name=api_name, storage_dir=f"models--Systran--faster-whisper-{suffix}")

    def matches_directory(self, directory: Path) -> bool:
        return directory.is_dir() and directory.name.lower() == self.storage_dir.lower()


class ModelDescriptor(Enum):
    TINY = ModelInfo.with_repo_suffix("tiny")
    TINY_EN = ModelInfo.with_repo_suffix("tiny.en")
    BASE = ModelInfo.with_repo_suffix("base")
    BASE_EN = ModelInfo.with_repo_suffix("base.en")
    SMALL = ModelInfo.with_repo_suffix("small")
    SMALL_EN = ModelInfo.with_repo_suffix("small.en")
    MEDIUM = ModelInfo.with_repo_suffix("medium")
    MEDIUM_EN = ModelInfo.with_repo_suffix("medium.en")
    LARGE = ModelInfo.with_repo_suffix("large", "large-v3")
    LARGE_V1 = ModelInfo.with_repo_suffix("large-v1")
    LARGE_V2 = ModelInfo.with_repo_suffix("large-v2")
    LARGE_V3 = ModelInfo.with_repo_suffix("large-v3")


MODEL_REGISTRY: Dict[str, ModelInfo] = {descriptor.value.api_name: descriptor.value for descriptor in ModelDescriptor}


class ModelRegistry:
    Tiny = ModelDescriptor.TINY.value
    TinyEn = ModelDescriptor.TINY_EN.value
    Base = ModelDescriptor.BASE.value
    BaseEn = ModelDescriptor.BASE_EN.value
    Small = ModelDescriptor.SMALL.value
    SmallEn = ModelDescriptor.SMALL_EN.value
    Medium = ModelDescriptor.MEDIUM.value
    MediumEn = ModelDescriptor.MEDIUM_EN.value
    Large = ModelDescriptor.LARGE.value
    LargeV1 = ModelDescriptor.LARGE_V1.value
    LargeV2 = ModelDescriptor.LARGE_V2.value
    LargeV3 = ModelDescriptor.LARGE_V3.value

    @classmethod
    def by_name(cls, api_name: str) -> ModelInfo:
        return MODEL_REGISTRY[api_name]


__all__ = ["ModelInfo", "ModelDescriptor", "ModelRegistry", "MODEL_REGISTRY"]
