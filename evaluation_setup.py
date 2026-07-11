"""Build reproducible, isolated corpora for offline evaluations."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence

from qa_system import MedicalQASystem
from vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_DOCUMENTS = (
    PROJECT_ROOT / "sample_docs" / "clinical_note_2024_03_15.txt",
    PROJECT_ROOT / "sample_docs" / "follow_up_note_2024_03_29.txt",
    PROJECT_ROOT / "data" / "sample_documents" / "sample_medical_record.txt",
)


@contextmanager
def isolated_evaluation_system(
    documents: Sequence[Path] = EVALUATION_DOCUMENTS,
) -> Iterator[MedicalQASystem]:
    """Yield a QA system backed by a fresh, temporary Chroma collection."""
    missing_documents = [str(path) for path in documents if not path.is_file()]
    if missing_documents:
        raise FileNotFoundError(
            "Evaluation corpus is incomplete: " + ", ".join(missing_documents)
        )

    with TemporaryDirectory(prefix="medhub-evaluation-") as persist_directory:
        vector_store = VectorStore(
            collection_name="medical_documents_evaluation",
            persist_directory=persist_directory,
        )
        qa = MedicalQASystem(vector_store=vector_store)
        for path in documents:
            qa.add_document(str(path))
        yield qa


def assert_labeled_chunks_present(qa: MedicalQASystem, cases: Sequence[dict]) -> None:
    """Fail before scoring when a retrieval label is absent from the corpus."""
    expected_ids = {
        chunk_id
        for case in cases
        for chunk_id in case.get("expected_chunk_ids", [])
    }
    stored_ids = set(qa.vector_store.collection.get().get("ids", []))
    missing_ids = sorted(expected_ids - stored_ids)
    if missing_ids:
        raise ValueError(
            "Retrieval labels reference chunks missing from the evaluation corpus: "
            + ", ".join(missing_ids)
        )
