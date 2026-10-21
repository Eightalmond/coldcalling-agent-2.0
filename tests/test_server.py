"""Offline tests for calle.server's HTTP API (fake LLM, no WebRTC, no network)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from calle.config import Settings
from calle.guardrails import (
    BRIEFS_PER_VISITOR_PER_HOUR,
    CALL_MAX_SECS,
    CALLS_PER_VISITOR_PER_HOUR,
    CallRegistry,
)
from calle.prompts import DEMO_BRIEF
from calle.server import create_app

VALID_BRIEF = DEMO_BRIEF.strip()
SETTINGS = Settings(deepgram_api_key="dg", groq_api_key="gq", tavily_api_key=None)


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        return VALID_BRIEF


async def no_call(connection: object, brief: str) -> None:
    raise AssertionError("no WebRTC call should start in these tests")


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(llm: FakeLLM) -> TestClient:
    app = create_app(SETTINGS, brief_llm=llm, call_runner=no_call, registry=CallRegistry())
    return TestClient(app)


def product_form(**extra: str) -> dict[str, str]:
    return {"product_name": "Quillo", "company_name": "Brightfern Labs", **extra}


def test_config_reports_limits_and_web_availability(client: TestClient) -> None:
    config = client.get("/api/config").json()
    assert config["call_max_secs"] == CALL_MAX_SECS
    assert config["calls_per_hour"] == CALLS_PER_VISITOR_PER_HOUR
    assert config["web_search_available"] is False


def test_demo_brief(client: TestClient) -> None:
    assert client.get("/api/demo-brief").json()["brief"] == VALID_BRIEF


def test_brief_from_pasted_text(client: TestClient, llm: FakeLLM) -> None:
    response = client.post("/api/briefs", data=product_form(text="Quillo takes notes."))
    assert response.status_code == 200
    assert response.json()["brief"] == VALID_BRIEF and llm.calls == 1


def test_brief_from_uploaded_markdown(client: TestClient) -> None:
    material = Path("examples/quillo_material.md").read_bytes()
    response = client.post(
        "/api/briefs",
        data=product_form(),
        files={"file": ("quillo.md", material, "text/markdown")},
    )
    assert response.status_code == 200


def test_brief_rejects_unsupported_file(client: TestClient, llm: FakeLLM) -> None:
    response = client.post(
        "/api/briefs", data=product_form(), files={"file": ("deck.pptx", b"x", "application/zip")}
    )
    assert response.status_code == 422 and "Unsupported file type" in response.json()["detail"]
    assert llm.calls == 0


def test_brief_requires_material(client: TestClient) -> None:
    response = client.post("/api/briefs", data=product_form())
    assert response.status_code == 422


def test_web_requested_without_key_warns(client: TestClient) -> None:
    response = client.post("/api/briefs", data=product_form(text="notes", use_web="true"))
    assert any("TAVILY_API_KEY" in w for w in response.json()["warnings"])


def test_brief_generation_is_rate_limited(client: TestClient) -> None:
    for _ in range(BRIEFS_PER_VISITOR_PER_HOUR):
        assert client.post("/api/briefs", data=product_form(text="notes")).status_code == 200
    response = client.post("/api/briefs", data=product_form(text="notes"))
    assert response.status_code == 429 and "Retry-After" in response.headers


def test_call_with_invalid_brief_lists_problems(client: TestClient) -> None:
    response = client.post("/api/calls", json={"brief": "## Overview\nToo short."})
    assert response.status_code == 422
    assert any("Not covered" in p for p in response.json()["problems"])


def test_call_created_with_valid_brief(client: TestClient) -> None:
    response = client.post("/api/calls", json={"brief": VALID_BRIEF})
    assert response.status_code == 200
    body = response.json()
    assert body["call_id"] and body["max_call_secs"] == CALL_MAX_SECS


def test_calls_are_rate_limited_per_visitor() -> None:
    app = create_app(SETTINGS, call_runner=no_call, registry=CallRegistry(max_concurrent=100))
    client = TestClient(app)
    for _ in range(CALLS_PER_VISITOR_PER_HOUR):
        assert client.post("/api/calls", json={"brief": VALID_BRIEF}).status_code == 200
    response = client.post("/api/calls", json={"brief": VALID_BRIEF})
    assert response.status_code == 429 and "this hour" in response.json()["detail"]


class FakeWebRTC:
    """Stands in for SmallWebRTCRequestHandler: records offers, no real WebRTC."""

    def __init__(self) -> None:
        self.offers: list[object] = []

    async def handle_web_request(self, request: object, webrtc_connection_callback: object) -> dict:
        self.offers.append(request)
        return {"sdp": "answer", "type": "answer", "pc_id": "pc-1"}

    async def close(self) -> None:
        pass


def test_offer_with_call_id_in_request_data_is_accepted() -> None:
    # The JS client sends the call id as camelCase `requestData`.
    webrtc = FakeWebRTC()
    client = TestClient(create_app(SETTINGS, call_runner=no_call, webrtc=webrtc))  # type: ignore[arg-type]
    call_id = client.post("/api/calls", json={"brief": VALID_BRIEF}).json()["call_id"]
    offer = {"sdp": "v=0", "type": "offer", "pc_id": None, "requestData": {"call_id": call_id}}
    response = client.post("/api/offer", json=offer)
    assert response.status_code == 200 and response.json()["pc_id"] == "pc-1"
    assert len(webrtc.offers) == 1
    # The same call id can't be used twice.
    assert client.post("/api/offer", json=offer).status_code == 403


def test_offer_without_valid_call_id_is_rejected(client: TestClient) -> None:
    offer = {"sdp": "v=0", "type": "offer"}
    assert client.post("/api/offer", json=offer).status_code == 403
    bad = client.post("/api/offer", json={**offer, "requestData": {"call_id": "made-up"}})
    assert bad.status_code == 403


def test_ice_candidates_for_unknown_connection_are_rejected(client: TestClient) -> None:
    response = client.patch("/api/offer", json={"pc_id": "nope", "candidates": []})
    assert response.status_code == 404
