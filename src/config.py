"""Configuration loader for Longspear.

Reads settings.yaml and persona files. Uses pydantic for validation.
Environment variables override YAML values via pydantic-settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Paths ─────────────────────────────────────────────────

def _project_root() -> Path:
    """Resolve project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def _config_dir() -> Path:
    return _project_root() / "config"


# ── Sub-models ────────────────────────────────────────────

class ChannelConfig(BaseModel):
    url: str
    name: str
    description: str = ""
    slug: str


class EmbeddingModelConfig(BaseModel):
    dimensions: int
    description: str = ""


class EmbeddingConfig(BaseModel):
    default: str
    models: dict[str, EmbeddingModelConfig]


class VectorStoreBackendConfig(BaseModel):
    description: str = ""
    path: str = ""


class VectorStoreConfig(BaseModel):
    default: str
    backends: dict[str, VectorStoreBackendConfig]


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100


class RetrievalConfig(BaseModel):
    top_k: int = 10
    score_threshold: float = 0.3


class PersonaConfig(BaseModel):
    name: str
    slug: str
    system_prompt: str
    speaking_style: dict[str, Any] = Field(default_factory=dict)
    biographical_context: str = ""


# ── Main Settings ─────────────────────────────────────────

class Settings(BaseSettings):
    """Main application settings loaded from YAML + env vars."""

    # Loaded from YAML
    data_cutoff_date: str = "2025-08-01"
    channels: dict[str, ChannelConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig | None = None
    vectorstore: VectorStoreConfig | None = None
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # Environment variables
    postgres_user: str = "longspear"
    postgres_password: str = "longspear_dev"
    postgres_db: str = "longspear"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    ollama_host: str = "http://localhost:11434"
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def data_dir(self) -> Path:
        return _project_root() / "data"

    def get_embedding_dims(self, model_name: str) -> int:
        """Get embedding dimensions for a given model name."""
        if self.embedding and model_name in self.embedding.models:
            return self.embedding.models[model_name].dimensions
        raise ValueError(f"Unknown embedding model: {model_name}")


def _load_yaml_settings() -> dict[str, Any]:
    """Load settings from config/settings.yaml."""
    config_path = _config_dir() / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings. Merges YAML + env vars."""
    yaml_data = _load_yaml_settings()
    return Settings(**yaml_data)


def load_persona(slug: str) -> PersonaConfig:
    """Load a persona config from config/personas/{slug}.yaml."""
    persona_path = _config_dir() / "personas" / f"{slug}.yaml"
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona not found: {persona_path}")
    with open(persona_path) as f:
        data = yaml.safe_load(f)
    return PersonaConfig(**data)


def list_personas() -> list[str]:
    """List available persona slugs."""
    personas_dir = _config_dir() / "personas"
    if not personas_dir.exists():
        return []
    return [p.stem for p in personas_dir.glob("*.yaml")]
