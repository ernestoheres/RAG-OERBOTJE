import json
from threading import Event

import pytest

import inference


@pytest.fixture
def client(monkeypatch):
    class FakePipeline:
        def retrieve(self, question):
            assert question == "Hoeveel EC?"
            return [{"blok": 12, "tekst": "De propedeuse heeft 60 EC.", "score": 1.0}]

        def stream_answer(self, question, hits, cancelled):
            yield "De propedeuse "
            yield "heeft 60 EC. [blok 12]"

        def answer(self, question):
            hits = self.retrieve(question)
            return {"answer": "".join(self.stream_answer(question, hits, Event())), "sources": hits}

    monkeypatch.setattr(inference, "_pipeline", FakePipeline())
    inference.app.config.update(TESTING=True)
    return inference.app.test_client()


def test_stream_contains_sources_and_incremental_answer(client):
    response = client.post("/api/chat/stream", json={"question": "Hoeveel EC?"})
    events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
    assert response.status_code == 200
    assert next(event for event in events if event["type"] == "sources")["sources"][0]["blok"] == 12
    assert [event["text"] for event in events if event["type"] == "token"] == [
        "De propedeuse ", "heeft 60 EC. [blok 12]",
    ]
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize("payload", [{}, {"question": "   "}, {"question": 123}, {"question": "x" * 2001}])
def test_invalid_questions_are_rejected(client, payload):
    assert client.post("/api/chat/stream", json=payload).status_code == 400
    assert client.post("/inference", json=payload).status_code == 400


def test_existing_json_endpoint_and_pdf_remain_available(client):
    result = client.post("/inference", json={"question": "Hoeveel EC?"}).get_json()
    assert result["answer"] == "De propedeuse heeft 60 EC. [blok 12]"
    assert result["sources"][0]["blok"] == 12
    pdf = client.get("/bron")
    assert pdf.status_code == 200
    assert pdf.mimetype == "application/pdf"


def test_stream_reports_generation_failure(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("model error")
        yield

    monkeypatch.setattr(inference._pipeline, "stream_answer", fail)
    response = client.post("/api/chat/stream", json={"question": "Hoeveel EC?"})
    events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
    assert events[-1]["type"] == "error"
    assert "model error" not in events[-1]["message"]
