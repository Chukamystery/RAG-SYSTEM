from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PHASE 2: EMBEDDINGS AND SEMANTIC SEARCH
# ==========================================

# Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# SMALL TEXT CORPUS
# ==========================================

documents = [
    "The customer requested a refund.",
    "The buyer wants their money back.",
    "Employees can work remotely on Fridays.",
    "Staff members may work from home on Friday.",
    "The company provides health insurance.",
    "Workers receive medical insurance.",
    "The office opens at 8 AM.",
    "Employees must wear their ID cards.",
]


# ==========================================
# CREATE DOCUMENT EMBEDDINGS
# ==========================================

embeddings = model.encode(documents)

print("Number of documents:", len(documents))
print("Embedding size:", len(embeddings[0]))


# ==========================================
# QUERY
# ==========================================

query = "How can I get my money back?"

print("\nQuery:")
print(query)


# ==========================================
# SEMANTIC SEARCH
# ==========================================

query_embedding = model.encode([query])

semantic_scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]


semantic_results = sorted(
    zip(documents, semantic_scores),
    key=lambda x: x[1],
    reverse=True
)


print("\nSemantic Search Results:")

for document, score in semantic_results:
    print(f"{score:.3f} - {document}")


# ==========================================
# SIMPLE WORD OVERLAP
# ==========================================

def word_overlap(query, document):
    query_words = set(query.lower().split())
    document_words = set(document.lower().split())

    common_words = query_words.intersection(document_words)

    return len(common_words)


overlap_results = []

for document in documents:
    score = word_overlap(query, document)
    overlap_results.append((document, score))


overlap_results = sorted(
    overlap_results,
    key=lambda x: x[1],
    reverse=True
)


print("\nWord Overlap Results:")

for document, score in overlap_results:
    print(f"{score} matching words - {document}")

    # ==========================================
# SAVE RESULTS FOR REPORT
# ==========================================

with open("phase2_results.txt", "w", encoding="utf-8") as file:

    file.write("PHASE 2: EMBEDDINGS AND SEMANTIC SEARCH\n\n")

    file.write(f"Query: {query}\n\n")

    file.write("SEMANTIC SEARCH RESULTS:\n")

    for document, score in semantic_results:
        file.write(f"{score:.3f} - {document}\n")

    file.write("\nWORD OVERLAP RESULTS:\n")

    for document, score in overlap_results:
        file.write(f"{score} matching words - {document}\n")

print("\nResults saved to phase2_results.txt")