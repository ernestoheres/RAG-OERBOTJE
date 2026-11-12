import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer

    return (SentenceTransformer,)


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = [
        "Hugging Face makes NLP easy.",
        "Transformers are powerful for deep learning."
    ]
    embeddings = model.encode(sentences)
    print("Embedding shape:", embeddings.shape)
    return (embeddings,)


@app.cell
def _(embeddings):
    embeddings
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
