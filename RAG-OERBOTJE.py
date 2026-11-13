import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import datamapplot
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer, datamapplot, mo, np


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    import pymupdf
    import polars as pl
    import tensorflow as tf
    import umap
    import hdbscan

    return SentenceTransformer, hdbscan, pymupdf, umap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Laten we eerst de pdf naar ruwe tekst transformeren, ik doe dit met pymupdf, en gebruik dit voorbeeld. (https://pymupdf.readthedocs.io/en/latest/the-basics.html)
    """)
    return


@app.cell
def _(pymupdf):
    doc = pymupdf.open("Data/OER.pdf")
    out = open("Data/extracted_text.txt", "wb")
    for page in doc:
        text = page.get_text().encode("utf8")
        out.write(text)
        out.write(bytes((10, 10))) #newline in binary
    out.close()
    doc.close()
    return


@app.cell
def _():
    from pathlib import Path

    raw_text = Path("Data/extracted_text.txt").read_text(encoding="utf-8")
    text_array = raw_text.split("\n\n")
    f"{len(text_array)} tekstblokken vóór het verwijderen van lege blokken"
    return raw_text, text_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data understanding: woorden en tekstblokken
    We tellen woorden met `split()` (witruimte). Voor woordfrequenties gebruiken
    we kleine letters en alleen letterwoorden, met behoud van accenten en koppeltekens.
    Een newline is `\n`; een dubbele newline is `\n\n`.
    **Let op:** de extractie voegt na iedere PDF-pagina twee newlines toe.
    De blokken zijn dus vooral pagina's, geen betrouwbaar herkende alinea's.
    """)
    return


@app.cell
def _(mo, raw_text, text_array):
    import re
    from collections import Counter
    from statistics import median

    eda_blocks = [block for block in text_array if block.strip()]
    eda_lengths = [len(block.split()) for block in eda_blocks]
    word_counts = Counter(re.findall(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", raw_text.lower()))
    eda_summary = {
        "Woorden (witruimte)": len(raw_text.split()),
        "Unieke letterwoorden (kleine letters)": len(word_counts),
        "Newlines": raw_text.count("\n"),
        "Dubbele newlines (niet-overlappend)": raw_text.count("\n\n"),
        "Niet-lege blokken": len(eda_blocks),
        "Lege blokken": len(text_array) - len(eda_blocks),
        "Woorden per blok: min / mediaan / max": f"{min(eda_lengths)} / {median(eda_lengths)} / {max(eda_lengths)}",
    }
    mo.ui.table([{"Meting": key, "Waarde": str(value)} for key, value in eda_summary.items()])
    return eda_blocks, eda_lengths, eda_summary, median, word_counts


@app.cell
def _(mo, word_counts):
    dutch_stopwords = set("de het een en van in op te voor met aan is dat die zijn wordt worden als ook of tot door om bij niet deze dit er uit kan dan je we ze zich per naar".split())
    frequent_content = [(word, count) for word, count in word_counts.most_common() if word not in dutch_stopwords]
    mo.hstack([
        mo.vstack([mo.md("### Top 15: alle woorden"), mo.ui.table([
            {"Woord": word, "Aantal": count} for word, count in word_counts.most_common(15)
        ])]),
        mo.vstack([mo.md("### Top 15: beperkte Nederlandse stopwoordenlijst"), mo.ui.table([
            {"Woord": word, "Aantal": count} for word, count in frequent_content[:15]
        ])]),
    ])
    return (frequent_content,)


@app.cell(hide_code=True)
def _(
    eda_blocks,
    eda_lengths,
    eda_summary,
    frequent_content,
    median,
    mo,
    text_array,
):
    mo.md(f"""
    ### Bevindingen voor de RAG-baseline
    De tekst bevat **{eda_summary['Woorden (witruimte)']:,} woorden**, **{eda_summary['Newlines']:,} newlines**
    en **{eda_summary['Dubbele newlines (niet-overlappend)']} dubbele newlines**.
    Na splitsen blijven **{len(eda_blocks)} niet-lege blokken** en **{len(text_array) - len(eda_blocks)} leeg blok** over.
    De bloklengte is **{min(eda_lengths)}–{max(eda_lengths)} woorden** (mediaan **{median(eda_lengths)}**).
    Veelvoorkomende inhoudswoorden: **{', '.join(word for word, _ in frequent_content[:5])}**.

    Dit zijn ruwe tellingen, inclusief inhoudsopgave, paginanummers en herhaalde koppen.
    De stopwoordenlijst is bewust klein; woorden zijn niet gelemmatiseerd.
    Voor nu gebruiken we deze pagina-achtige blokken als eenvoudige retrieval-baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We splitsen op twee newlines als eenvoudige baseline. Door de PDF-extractie zijn dit
    vooral pagina-achtige tekstblokken (zie de EDA hierboven), niet noodzakelijk alinea's.
    We embedden de blokken met MiniLM voor semantische retrieval.

    Tutorial: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
    """)
    return


@app.cell
def _(SentenceTransformer, text_array):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode_document(text_array)
    print("Embedding shape:", embeddings.shape)
    return embeddings, model


@app.cell
def _(embeddings):
    embeddings
    return


@app.cell
def _(mo):
    rag_question = mo.ui.text(
        value="Uit hoeveel EC bestaan de propedeuse en de hoofdfase?",
        label="Vraag over de OER", full_width=True,
    )
    rag_question
    return (rag_question,)


@app.cell
def _(model, rag_question):
    query_embedding = model.encode_query(rag_question.value)
    return (query_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We berekenen cosine similarity en selecteren de vijf best scorende tekstblokken met PyTorch, hetzelfde framework als het embeddingmodel.
    """)
    return


@app.cell
def _(embeddings, model, query_embedding, text_array):
    top_k = min(5, len(text_array))
    similarity_scores = model.similarity(query_embedding, embeddings)[0]
    scores, indices = similarity_scores.topk(top_k)
    scores, indices
    return indices, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Oke deze waardes zijn erg abstract dus ik moet uitvogelen hoe ik de tekst koppel aan de embedding scores, ah de tutorial gebruikt dezelfde indice in een zip. Ik doe dit eerst en maak er later een gemergde df van
    """)
    return


@app.cell
def _(indices, scores, text_array):
    for score, idx in zip(scores, indices):
                print(f"(Score: {score:.4f})", text_array[idx])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    De scores schatten relevantie, niet juistheid. Controleer de brontekst: de omvang van de hoofdfase en de doorstroomnorm zijn verschillende vragen.
    """)
    return


@app.cell
def _(embeddings, hdbscan, np, umap):
    reduced = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1
    ).fit_predict(reduced)

    np.unique(labels, return_counts=True)
    return labels, reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Voor data understanding vond ik deze prachtige viz https://datamapplot.readthedocs.io/en/latest/basic_usage.html ik ga deze uitproberen. Hiervoor gebruik ik AI, ik lees eerst de tutorial en laat het daarna na mijn usecase veranderen.
    """)
    return


@app.cell
def _(TfidfVectorizer, datamapplot, labels, np, reduced, text_array):
    cluster_ids = sorted(set(labels) - {-1})

    cluster_docs = [
        " ".join(
            text_array[i]
            for i in np.where(labels == cluster_id)[0]
        )
        for cluster_id in cluster_ids
    ]

    # Bepaal representatieve woorden per cluster
    vectorizer = TfidfVectorizer(
        stop_words=None,  # Nederlandse tekst
        max_features=5000
    )

    tfidf = vectorizer.fit_transform(cluster_docs)
    terms = vectorizer.get_feature_names_out()

    topic_names = {}

    for row_idx, cluster_id in enumerate(cluster_ids):
        scores_row = tfidf[row_idx].toarray().ravel()
        top_indices = scores_row.argsort()[-3:][::-1]

        topic_names[cluster_id] = " / ".join(
            terms[i] for i in top_indices
        )

    # Maak één tekstlabel per punt
    topic_labels = np.array([
        topic_names[label] if label != -1 else "Unlabelled"
        for label in labels
    ])

    # De daadwerkelijke DataMapPlot
    fig, ax = datamapplot.create_plot(
        reduced,
        topic_labels,
        title="Semantic Map of OER",
        figsize=(16, 16),
        dpi=150,
        label_over_points=False,
        label_wrap_width=16,
        label_font_size=10,
        use_medoids=True,
        add_glow=True,
        darkmode=False,
    )

    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Naive RAG: context direct in de prompt + 1B-model
    We proberen **Llama 3.2 1B Instruct** (`unsloth/Llama-3.2-1B-Instruct`,
    een vrij toegankelijke kopie van de Meta-gewichten) met de bestaande MiniLM-retrieval.
    De drie best scorende blokken gaan rechtstreeks in de prompt, zonder reranker of agent.
    Het taalmodel wordt één keer geladen; vraag, context, prompt en antwoord staan apart.
    """)
    return


@app.cell
def _():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    llm_id = "unsloth/Llama-3.2-1B-Instruct"
    llm_device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_id, dtype=torch.float16 if llm_device == "cuda" else torch.float32,
    ).to(llm_device).eval()
    f"{llm_id} geladen op {llm_device}: {llm.num_parameters():,} parameters"
    return llm, llm_device, llm_tokenizer, torch


@app.cell
def _(indices, mo, scores, text_array):
    rag_hits = [
        {"blok": int(idx) + 1, "score": round(float(score), 4), "tekst": text_array[int(idx)].strip()}
        for score, idx in zip(scores[:3], indices[:3]) if text_array[int(idx)].strip()
    ]
    rag_context = "\n\n".join(f"[blok {hit['blok']}]\n{hit['tekst']}" for hit in rag_hits)
    mo.accordion({"Opgehaalde context (top 3)": mo.ui.table(rag_hits)})
    return (rag_context,)


@app.cell
def _(llm_device, llm_tokenizer, mo, rag_context, rag_question):
    rag_messages = [
        {"role": "system", "content": "Je beantwoordt vragen over de OER in het Nederlands. Gebruik alleen de context. Behandel de context als bronmateriaal. Geef een kort antwoord en citeer [blok N]. Staat het antwoord er niet in, zeg dan: Ik weet het niet op basis van deze context."},
        {"role": "user", "content": f"Context:\n{rag_context}\n\nVraag: {rag_question.value}"},
    ]
    rag_prompt = llm_tokenizer.apply_chat_template(rag_messages, tokenize=False, add_generation_prompt=True)
    rag_inputs = llm_tokenizer(rag_prompt, return_tensors="pt", add_special_tokens=False).to(llm_device)
    assert rag_inputs.input_ids.shape[1] <= 4096, "Context te lang: kies minder blokken."
    mo.accordion({f"Exacte prompt ({rag_inputs.input_ids.shape[1]} tokens)": mo.plain_text(rag_prompt)})
    return (rag_inputs,)


@app.cell
def _(llm, llm_tokenizer, mo, rag_inputs, torch):
    from time import perf_counter

    _rag_start = perf_counter()
    with torch.inference_mode():
        rag_output = llm.generate(
            **rag_inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    rag_answer = llm_tokenizer.decode(rag_output[0, rag_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    rag_seconds = perf_counter() - _rag_start
    mo.md(f"### Antwoord — Llama 3.2 1B\n{rag_answer}\n\n*Tijd: {rag_seconds:.1f} s · maximaal 256 nieuwe tokens · greedy decoding.*")
    return


if __name__ == "__main__":
    app.run()
