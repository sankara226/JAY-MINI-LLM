"""Tests for the server settings in app.py."""
import pytest

import app


def test_all_versions_by_default(monkeypatch):
    monkeypatch.delenv("JAY_VERSIONS", raising=False)
    assert app.allowed_versions() == ["v1", "v2", "v3"]


def test_jay_versions_limits_what_is_offered(monkeypatch):
    monkeypatch.setenv("JAY_VERSIONS", " v3 ")
    assert app.allowed_versions() == ["v3"]
    monkeypatch.setenv("JAY_VERSIONS", "v2,v3")
    assert app.allowed_versions() == ["v2", "v3"]


def test_jay_versions_rejects_typos(monkeypatch):
    monkeypatch.setenv("JAY_VERSIONS", "v4")
    with pytest.raises(ValueError, match="unknown versions"):
        app.allowed_versions()
