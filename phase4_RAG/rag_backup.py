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
# SEMANTIC SEARCH
# ==========================================

query = "What is the company's maternity leave policy?"

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

for result in top_results:
    print("-", result["filename"])