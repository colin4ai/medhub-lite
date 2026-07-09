"""Smoke tests: verify the app imports and core modules load."""

def test_api_imports():
    import api

def test_agents_imports():
    import agents

def test_router_prompt_exists():
    from agents import ROUTER_PROMPT
    assert "qa" in ROUTER_PROMPT and "extract" in ROUTER_PROMPT
