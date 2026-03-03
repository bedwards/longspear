"""Tests for the configuration system."""

from pathlib import Path

from src.config import (
    Settings,
    get_settings,
    list_personas,
    load_persona,
    _load_yaml_settings,
)


def test_load_yaml_settings():
    """YAML settings file loads correctly."""
    data = _load_yaml_settings()
    assert "data_cutoff_date" in data
    assert "channels" in data
    assert "embedding" in data
    assert "vectorstore" in data


def test_settings_defaults():
    """Settings has sensible defaults."""
    settings = get_settings()
    assert settings.data_cutoff_date == "2025-08-01"
    assert "heather_cox_richardson" in settings.channels
    assert "nate_b_jones" in settings.channels


def test_settings_embedding_dims():
    """Embedding dimensions are correct."""
    settings = get_settings()
    assert settings.get_embedding_dims("nomic-embed-text") == 768
    assert settings.get_embedding_dims("mxbai-embed-large") == 1024


def test_settings_postgres_dsn():
    """Postgres DSN is well-formed."""
    settings = get_settings()
    assert settings.postgres_dsn.startswith("postgresql://")


def test_list_personas():
    """Personas directory has expected files."""
    slugs = list_personas()
    assert "heather_cox_richardson" in slugs
    assert "nate_b_jones" in slugs


def test_load_persona():
    """Persona config loads with expected fields."""
    persona = load_persona("nate_b_jones")
    assert persona.name == "Nate B Jones"
    assert persona.slug == "nate_b_jones"
    assert persona.system_prompt
    assert persona.speaking_style
