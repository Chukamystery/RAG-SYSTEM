import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ollama

DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data"
)


# ==========================================
# READ DOCUMENTS
# ==========================================

documents = []

for filename in os.listdir(DATA_FOLDER):

    if filename.endswith(".txt"):

        filepath = os.path.join(DATA_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append({
            "filename": filename,
            "text": text
        })


print("Number of documents:", len(documents))


# ==========================================
# CREATE CHUNKS
# ==========================================

chunks = []

for document in documents:

    # Split document into paragraphs
    paragraphs = document["text"].split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:

           chunks.append({
    "chunk_id": len(chunks),
    "filename": document["filename"],
    "text": paragraph
})


# ==========================================
# DISPLAY CHUNKS
# ==========================================

print("\nNumber of chunks:", len(chunks))

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i + 1}")
    print("Source:", chunk["filename"])
    print("Text:", chunk["text"])

    # ==========================================
# CREATE EMBEDDINGS
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(chunk_texts)

print("\nEmbedding shape:", embeddings.shape)

# ==========================================
# PCA VISUALIZATION
# ==========================================

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)


embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))

for i, chunk in enumerate(chunks):

    x = embeddings_2d[i, 0]
    y = embeddings_2d[i, 1]

    plt.scatter(x, y)

    plt.annotate(
        chunk["filename"],
        (x, y),
        fontsize=8
    )

plt.title("2D PCA Visualization of Document Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()

plt.savefig("pca_embeddings.png")

plt.show(block=False)
plt.pause(2)
plt.close()

# ==========================================
# SEMANTIC SEARCH
# ==========================================

query = input("\nAsk a question: ")

query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    embeddings
)[0]

results = []

for i, score in enumerate(similarities):

    results.append({
        "score": score,
        "text": chunks[i]["text"],
        "filename": chunks[i]["filename"]
    })


results.sort(
    key=lambda x: x["score"],
    reverse=True
)


print("\nQuery:")
print(query)

print("\nSearch Results:")

for result in results[:3]:

    print(
        f'{result["score"]:.3f} - '
        f'{result["text"]} '
        f'(Source: {result["filename"]})'
    )

    # ==========================================
# GENERATE ANSWER WITH OLLAMA
# ==========================================

top_results = results[:3]

context = "\n\n".join(
    result["text"] for result in top_results
)

prompt = f"""
You are a helpful assistant answering questions from a company knowledge base.

Use ONLY the information provided in the context.

Do not add, assume, or change any information.

If the context does not contain the answer, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{query}

Answer:
"""

response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "num_ctx": 1024
    }
)

answer = response["message"]["content"]

print("\nGenerated Answer:")
print(answer)

print("\nSources:")

unique_sources = []

for result in top_results:
    if result["filename"] not in unique_sources:
        unique_sources.append(result["filename"])

for source in unique_sources:
    print("-", source)