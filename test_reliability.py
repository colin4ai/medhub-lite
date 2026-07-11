"""Offline regression tests for reliability controls."""
from types import SimpleNamespace

import config
from agents import AgentOrchestrator
from evaluator import QAEvaluator
from qa_system import MedicalQASystem
from vector_store import VectorStore


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5):
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
    qa = SimpleNamespace(ask_question=lambda query, top_k: {"answer": "ok"})
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


def test_vector_store_is_persistent(tmp_path):
    first = VectorStore(collection_name="persistence_test", persist_directory=str(tmp_path))
    first.collection.upsert(
        ids=["chunk"], documents=["evidence"], metadatas=[{"doc_id": "doc"}],
        embeddings=[[0.0, 1.0]],
    )
    second = VectorStore(collection_name="persistence_test", persist_directory=str(tmp_path))
    assert second.get_stats()["total_chunks"] == 1


def test_lexical_reranker_ignores_common_question_words():
    assert VectorStore._lexical_score(
        "What allergy is recorded?", "ALLERGIES: Penicillin causes rash"
    ) > VectorStore._lexical_score(
        "What allergy is recorded?", "The patient attended physical therapy"
    )
