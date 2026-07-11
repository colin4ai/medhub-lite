# MedHub Lite quick start

## Local setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Add OPENAI_API_KEY and a long random API_AUTH_KEY to .env.
python -m pytest -q
```

## CLI demo

```bash
python main.py index --input sample_docs --reset
python main.py ask "What are the patient's current work restrictions?"
python main.py stats
```

The default CLI tenant is `default`. Re-index after upgrading a pre-tenancy collection,
or use `migrate_tenant_metadata.py` as described in `AWS_DEPLOYMENT.md`.

## API demo

```bash
python api.py
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health/ready

curl -X POST http://127.0.0.1:8000/upload \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "X-Tenant-ID: demo" \
  -F "file=@sample_docs/clinical_note_2024_03_15.txt"

curl -X POST http://127.0.0.1:8000/ask \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "X-Tenant-ID: demo" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the primary diagnosis?","top_k":5}'
```

## Evaluation

```bash
python evaluate_routing.py
python evaluate_retrieval.py
python evaluate_refusal.py
python main.py evaluate --test-set test_cases.json
```

The evaluation sets are small development/regression sets. They demonstrate failure-mode
coverage, not production accuracy. Model comparison makes paid calls:

```bash
python evaluate_models.py --models gpt-4o-mini gpt-4o
```

## Defensible interview description

MedHub Lite is a production-oriented prototype, not a validated medical product. It
demonstrates:

- token-bounded document processing and metadata extraction;
- tenant-filtered hybrid retrieval;
- atomic claims and structured fields with exact source evidence;
- optional semantic entailment verification;
- abstention on unsupported answers;
- evaluated agentic routing and grounded specialist workflows;
- request, latency, token, and infrastructure observability;
- containerization and a validated AWS reference stack.

Current constraints should be stated directly: the datasets are synthetic and small,
embedded Chroma is limited to one service task, tenant API keys are a demo authentication
mechanism, multimodal processing is not implemented, and no real user-impact study has
been conducted.

See `README.md` for design details and `AWS_DEPLOYMENT.md` for deployment.
