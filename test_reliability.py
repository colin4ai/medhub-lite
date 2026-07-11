"""Offline regression tests for reliability controls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from agents import AgentOrchestrator
from evaluation_setup import EVALUATION_DOCUMENTS, assert_labeled_chunks_present
from evaluator import QAEvaluator
from qa_system import MedicalQASystem
from vector_store import VectorStore
from document_processor import DocumentProcessor, MedicalDocument


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5, filter_metadata=None):
        return self.chunks[:top_k]


def make_qa(chunks, generated):
    qa = MedicalQASystem.__new__(MedicalQASystem)
    qa.vector_store = FakeStore(chunks)
    qa._generate_answer = lambda question, context: generated
    return qa


def test_invalid_citation_forces_refusal():
    chunks = [{"content": "Documented diagnosis", "metadata": {"filename": "a.txt"},
               "chunk_id": "a_0", "similarity": 0.9}]
    qa = make_qa(chunks, {"answer": "Unsupported", "answerable": True,
                          "cited_source_numbers": [99]})
    result = qa.ask_question("diagnosis?")
    assert result["answerable"] is False
    assert result["confidence"] == "low"


def test_confidence_uses_evidence_similarity():
    chunks = [{"content": "Documented diagnosis", "metadata": {"filename": "a.txt"},
               "chunk_id": "a_0", "similarity": 0.9}]
    qa = make_qa(chunks, {"answer": "Diagnosis [Source 1]", "answerable": True,
                          "cited_source_numbers": [1]})
    result = qa.ask_question("diagnosis?")
    assert result["answerable"] is True
    assert result["confidence"] == "high"


def test_claim_requires_exact_source_quote():
    chunks = [{"content": "MRI showed disc herniation at L4-L5.", "metadata": {},
               "chunk_id": "a_0", "similarity": 0.9}]
    answer, cited, valid, evidence = MedicalQASystem._validate_claims([
        {"text": "The MRI showed an L4-L5 herniation.", "evidence": [
            {"source_number": 1, "quote": "disc herniation at L4-L5"}
        ]}
    ], chunks)
    assert valid is True
    assert cited == {1}
    assert "[Source 1]" in answer
    assert evidence[0]["evidence"][0]["quote"] == "disc herniation at L4-L5"


def test_claim_rejects_quote_not_present_in_source():
    chunks = [{"content": "MRI showed disc herniation.", "metadata": {},
               "chunk_id": "a_0", "similarity": 0.9}]
    _, _, valid, _ = MedicalQASystem._validate_claims([
        {"text": "Surgery is required.", "evidence": [
            {"source_number": 1, "quote": "surgery is definitely required"}
        ]}
    ], chunks)
    assert valid is False


def test_evaluator_checks_actual_citation_ids():
    assert QAEvaluator._valid_citations({
        "cited_source_numbers": [1], "sources": [{"source_number": 1}]
    })
    assert not QAEvaluator._valid_citations({
        "cited_source_numbers": [2], "sources": [{"source_number": 1}]
    })
    assert QAEvaluator._normalize("light-duty") == QAEvaluator._normalize("light duty")


def test_invalid_agent_route_falls_back_to_qa(monkeypatch):
    qa = SimpleNamespace(ask_question=lambda query, top_k, tenant_id: {"answer": "ok"})
    orchestrator = AgentOrchestrator(qa, FakeStore([]))
    monkeypatch.setattr(orchestrator, "_llm", lambda *args, **kwargs: '{"route":"delete"}')
    result = orchestrator.run("question")
    assert result["route"] == "qa"
    assert result["result"]["answer"] == "ok"


def test_extraction_requires_evidence_for_each_populated_field():
    data = {
        "patient_name": "John Doe", "date_of_visit": None, "provider": None,
        "chief_complaint": None, "diagnoses": [], "medications": [], "plan": [],
    }
    chunks = [{"content": "Patient Name: John Doe"}]
    assert AgentOrchestrator._validate_extraction_evidence(data, [
        {"field": "patient_name", "source_number": 1, "quote": "Patient Name: John Doe"}
    ], chunks)
    assert not AgentOrchestrator._validate_extraction_evidence(data, [], chunks)


def test_retrieval_evaluator_calculates_recall_at_k():
    chunks = [{"content": "x", "metadata": {}, "chunk_id": "relevant", "similarity": 0.8}]
    qa = SimpleNamespace(vector_store=FakeStore(chunks))
    report = QAEvaluator(qa).evaluate_retrieval_quality([
        {"question": "q", "expected_chunk_ids": ["relevant", "missing"]}
    ])
    assert report["mean_recall_at_k"] == 0.5


def test_evaluation_corpus_contains_every_labeled_document():
    expected_doc_ids = {
        chunk_id.rsplit("_chunk_", 1)[0].split(":", 1)[-1]
        for case in json.loads(
            (Path(__file__).parent / "retrieval_cases.json").read_text()
        )
        for chunk_id in case["expected_chunk_ids"]
    }
    assert expected_doc_ids.issubset({path.stem for path in EVALUATION_DOCUMENTS})


def test_retrieval_evaluation_rejects_missing_labeled_chunks():
    collection = SimpleNamespace(get=lambda: {"ids": ["default:present_chunk_0"]})
    qa = SimpleNamespace(vector_store=SimpleNamespace(collection=collection))
    with pytest.raises(ValueError, match="default:missing_chunk_0"):
        assert_labeled_chunks_present(qa, [
            {"expected_chunk_ids": ["default:present_chunk_0", "default:missing_chunk_0"]}
        ])


def test_vector_store_is_persistent(tmp_path):
    first = VectorStore(collection_name="persistence_test", persist_directory=str(tmp_path))
    first.collection.upsert(
        ids=["chunk"], documents=["evidence"], metadatas=[{"doc_id": "doc"}],
        embeddings=[[0.0, 1.0]],
    )
    second = VectorStore(collection_name="persistence_test", persist_directory=str(tmp_path))
    assert second.get_stats()["total_chunks"] == 1


def test_reingestion_removes_stale_chunks_after_success(tmp_path):
    store = VectorStore(collection_name="replace_test", persist_directory=str(tmp_path))
    store.embedding_generator.generate_embeddings_batch = lambda texts: [[1.0, 0.0] for _ in texts]
    base = {"doc_id": "default:doc", "filename": "doc.txt", "tenant_id": "default"}
    store.add_chunks([{**base, "chunk_id": f"default:doc_chunk_{i}", "content": str(i)} for i in range(3)])
    store.add_chunks([{**base, "chunk_id": "default:doc_chunk_0", "content": "new"}])
    assert store.collection.get(where={"doc_id": "default:doc"})["ids"] == ["default:doc_chunk_0"]


def test_embedding_ingestion_is_batched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 2)
    store = VectorStore(collection_name="batch_test", persist_directory=str(tmp_path))
    batch_sizes = []
    store.embedding_generator.generate_embeddings_batch = lambda texts: (
        batch_sizes.append(len(texts)) or [[1.0, 0.0] for _ in texts]
    )
    base = {"doc_id": "default:doc", "filename": "doc.txt", "tenant_id": "default"}
    store.add_chunks([{**base, "chunk_id": f"default:doc_chunk_{i}", "content": str(i)} for i in range(5)])
    assert batch_sizes == [2, 2, 1]


def test_lexical_reranker_ignores_common_question_words():
    assert VectorStore._lexical_score(
        "What allergy is recorded?", "ALLERGIES: Penicillin causes rash"
    ) > VectorStore._lexical_score(
        "What allergy is recorded?", "The patient attended physical therapy"
    )


def test_long_paragraph_is_split_within_token_budget():
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    document = MedicalDocument("word " * 300, {
        "doc_id": "default:long", "filename": "long.txt", "tenant_id": "default"
    })
    chunks = processor.chunk_document(document)
    assert len(chunks) > 1
    assert all(len(processor.tokenizer.encode(chunk["content"])) <= 50 for chunk in chunks)


def test_many_short_paragraphs_include_separator_tokens_in_budget():
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=4)
    document = MedicalDocument("\n\n".join(["short paragraph"] * 20), {
        "doc_id": "default:paragraphs", "filename": "paragraphs.txt", "tenant_id": "default"
    })
    chunks = processor.chunk_document(document)
    assert all(len(processor.tokenizer.encode(chunk["content"])) <= 20 for chunk in chunks)
