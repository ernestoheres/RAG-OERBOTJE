"""Serve the notebook's RAG pipeline: python inference.py.

POST /inference with JSON {"question": "..."} to receive an answer and sources.
"""

from functools import lru_cache
from pathlib import Path
from threading import Lock
import tomllib

import pymupdf
import torch
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


app = Flask(__name__)
_inference_lock = Lock()
SYSTEM_PROMPT = (
    "Je bent een behulpzame Nederlandstalige assistent voor vragen over de OER. "
    "Lees de bron zorgvuldig en beantwoord de vraag met de informatie uit de bron. "
    "Noem de relevante feiten en getallen en verwijs naar het bronblok met [blok N]. "
    "Als de bron het antwoord echt niet bevat, geef dan aan dat je het niet weet. "
    "Begroetingen mag je vriendelijk beantwoorden zonder de bron te gebruiken."
)


def load_text_blocks():
    pdf_path = Path(__file__).resolve().parent / "Data" / "OER.pdf"
    with pymupdf.open(pdf_path) as doc:
        raw_text = "".join(page.get_text() + "\n\n" for page in doc)
    return raw_text.split("\n\n")


def load_config():
    config_path = Path(__file__).resolve().parent / "inference.toml"
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_language_model(config):
    llm_id = config["llm_id"]
    device = config["device"]
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError('device must be "auto", "cuda", or "cpu"')
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_id, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return tokenizer, llm, device


class RAGPipeline:
    def __init__(self):
        self.text_array = load_text_blocks()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = self.model.encode_document(self.text_array)
        self.tokenizer, self.llm, self.device = load_language_model(load_config())

    def retrieve(self, question):
        query_embedding = self.model.encode_query(question)
        similarity_scores = self.model.similarity(query_embedding, self.embeddings)[0]
        scores, indices = similarity_scores.topk(min(5, len(self.text_array)))
        return [
            {
                "blok": int(idx) + 1,
                "score": round(float(score), 4),
                "tekst": self.text_array[int(idx)].strip(),
            }
            for score, idx in zip(scores[:3], indices[:3])
            if self.text_array[int(idx)].strip()
        ]

    def generate(self, question, hits):
        context = "\n\n".join(f"[blok {hit['blok']}]\n{hit['tekst']}" for hit in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Bron:\n{context}\n\nVraag: {question}\n\nGeef een kort antwoord op de vraag op basis van de bron."},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        if inputs.input_ids.shape[1] > 4096:
            raise ValueError("Context te lang: kies minder blokken.")
        with torch.inference_mode():
            output = self.llm.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

    def answer(self, question):
        hits = self.retrieve(question)
        return {"answer": self.generate(question, hits), "sources": hits}


@lru_cache(maxsize=1)
def get_pipeline():
    return RAGPipeline()


def answer_question(question):
    with _inference_lock:
        return get_pipeline().answer(question)


@app.get("/")
def index():
    return render_template("chat.html")


@app.post("/chat")
def chat():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template("message.html", error="Vul een vraag in.")
    try:
        result = answer_question(question)
    except ValueError as exc:
        return render_template("message.html", question=question, error=str(exc))
    return render_template("message.html", question=question, result=result)


@app.post("/inference")
def inference():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("question"), str) or not data["question"].strip():
        return jsonify(error="Provide a non-empty question in the JSON body."), 400
    try:
        result = answer_question(data["question"])
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
