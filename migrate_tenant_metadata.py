"""Migrate pre-tenancy Chroma records to the default tenant.

Dry-run by default. Back up storage, stop writers, then use --apply exactly once.
"""
import argparse

import config
from qa_system import MedicalQASystem
from vector_store import VectorStore


def migrate(apply=False, tenant_id=config.DEFAULT_TENANT_ID, store=None):
    tenant_id = MedicalQASystem.normalize_tenant_id(tenant_id)
    store = store or VectorStore()
    result = store.collection.get(include=["documents", "metadatas", "embeddings"])
    candidates = []
    for index, old_id in enumerate(result.get("ids", [])):
        metadata = result["metadatas"][index] or {}
        if metadata.get("tenant_id"):
            continue
        new_metadata = {**metadata, "tenant_id": tenant_id}
        doc_id = new_metadata.get("doc_id", "unknown")
        new_metadata["doc_id"] = doc_id if doc_id.startswith(f"{tenant_id}:") else f"{tenant_id}:{doc_id}"
        candidates.append({
            "old_id": old_id,
            "new_id": old_id if old_id.startswith(f"{tenant_id}:") else f"{tenant_id}:{old_id}",
            "document": result["documents"][index],
            "metadata": new_metadata,
            "embedding": result["embeddings"][index],
        })
    if apply and candidates:
        store.collection.upsert(
            ids=[item["new_id"] for item in candidates],
            documents=[item["document"] for item in candidates],
            metadatas=[item["metadata"] for item in candidates],
            embeddings=[item["embedding"] for item in candidates],
        )
        old_ids = [item["old_id"] for item in candidates if item["old_id"] != item["new_id"]]
        if old_ids:
            store.collection.delete(ids=old_ids)
    return {"candidates": len(candidates), "applied": bool(apply), "tenant_id": tenant_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tenant", default=config.DEFAULT_TENANT_ID)
    args = parser.parse_args()
    print(migrate(args.apply, args.tenant))
