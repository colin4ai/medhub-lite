"""API authentication, request tracing, and tenant-isolation tests."""
from fastapi.testclient import TestClient

import config
from api import app
from qa_system import MedicalQASystem
from vector_store import VectorStore
from migrate_tenant_metadata import migrate


def test_metrics_requires_configured_api_key(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "test-secret")
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        response = client.get("/metrics", headers={"X-API-Key": "test-secret"})
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID")


def test_upload_rejects_unsupported_file_type(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("malware.exe", b"x")})
        assert response.status_code == 415


def test_tenant_identifier_validation():
    assert MedicalQASystem.normalize_tenant_id("Tenant_1") == "tenant_1"
    for invalid in ("", "../other", "spaces are bad", "a" * 65):
        try:
            MedicalQASystem.normalize_tenant_id(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid tenant ID: {invalid!r}")


def test_tenant_key_cannot_access_another_tenant(monkeypatch):
    monkeypatch.setattr(config, "TENANT_API_KEYS", {"alpha": "alpha-key", "beta": "beta-key"})
    with TestClient(app) as client:
        response = client.delete(
            "/documents", headers={"X-Tenant-ID": "beta", "X-API-Key": "alpha-key"}
        )
        assert response.status_code == 401


def test_global_metrics_require_a_valid_key_in_tenant_key_mode(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    monkeypatch.setattr(config, "TENANT_API_KEYS", {"alpha": "alpha-key"})
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"X-API-Key": "alpha-key"}).status_code == 200


def test_delete_tenant_does_not_delete_other_tenants(tmp_path):
    store = VectorStore(collection_name="tenant_test", persist_directory=str(tmp_path))
    store.collection.upsert(
        ids=["a", "b"], documents=["one", "two"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"tenant_id": "alpha"}, {"tenant_id": "beta"}],
    )
    assert store.delete_tenant("alpha") == 1
    assert store.collection.get(where={"tenant_id": "beta"})["ids"] == ["b"]


def test_legacy_records_can_be_migrated_without_reembedding(tmp_path):
    store = VectorStore(collection_name="migration_test", persist_directory=str(tmp_path))
    store.collection.upsert(
        ids=["doc_chunk_0"], documents=["legacy evidence"], embeddings=[[1.0, 0.0]],
        metadatas=[{"doc_id": "doc", "filename": "doc.txt"}],
    )
    result = migrate(apply=True, tenant_id="alpha", store=store)
    assert result["candidates"] == 1
    migrated = store.collection.get(ids=["alpha:doc_chunk_0"])
    assert migrated["metadatas"][0]["tenant_id"] == "alpha"
