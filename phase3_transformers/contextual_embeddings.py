# import the libraries
# AutoTokenizer is used to break the sentences into tokens.
# AutoModel loads the pretrained Transformer model.
# PyTorch is used to work with the model's numerical representations
# and calculate similarity.

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# ==========================================
# PHASE 3: TRANSFORMER CONTEXT EXPERIMENT
# ==========================================

# I loaded the pretrained DistilBERT model and its tokenizer
# so that I could process sentences and extract
# contextual representations from them.

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)


# ==========================================
# SENTENCES
# ==========================================

sentences = [
    "The bank approved my loan.",
    "I sat beside the river bank.",
    "The mouse ate some cheese.",
    "I connected the mouse to my computer."
    "The bat flew out of the cave.",
    "He hit the ball with the bat."
]


# ==========================================
# FUNCTION TO GET INITIAL AND FINAL
# REPRESENTATIONS OF A WORD
# ==========================================

def get_word_embedding(sentence, target_word):

    # I passed the sentence through the tokenizer,
    # which converted the text into tokens that
    # the Transformer could process.

    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )

    # Convert token IDs back into readable tokens
    # so that I can find the position of the target word.

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    # Find the position of the target word.

    target_positions = []

    for i, token in enumerate(tokens):

        if token == target_word:
            target_positions.append(i)

    if not target_positions:

        raise ValueError(
            f"Could not find '{target_word}' in: {sentence}"
        )

    position = target_positions[0]


    # ==========================================
    # INITIAL EMBEDDING
    # ==========================================

    # I extracted the initial embedding of the word
    # before it passes through the Transformer layers.

    with torch.no_grad():

        initial_embeddings = model.get_input_embeddings()(
            inputs["input_ids"]
        )

    initial_vector = initial_embeddings[0, position]


    # ==========================================
    # TRANSFORMER PROCESSING
    # ==========================================

    # I passed the tokenized sentence through DistilBERT.
    # The Transformer uses the surrounding context
    # to create contextual representations.

    with torch.no_grad():

        outputs = model(**inputs)


    # ==========================================
    # FINAL CONTEXTUAL REPRESENTATION
    # ==========================================

    # I extracted the final Transformer representation
    # of the target word.

    final_vector = outputs.last_hidden_state[0, position]


    return initial_vector, final_vector, tokens


# ==========================================
# GET BANK REPRESENTATIONS
# ==========================================

# Financial meaning of "bank"

bank_initial_1, bank_1, tokens_1 = get_word_embedding(
    sentences[0],
    "bank"
)


# River meaning of "bank"

bank_initial_2, bank_2, tokens_2 = get_word_embedding(
    sentences[1],
    "bank"
)


# ==========================================
# GET MOUSE REPRESENTATIONS
# ==========================================

# Animal meaning of "mouse"

mouse_initial_1, mouse_1, tokens_3 = get_word_embedding(
    sentences[2],
    "mouse"
)


# Computer-device meaning of "mouse"

mouse_initial_2, mouse_2, tokens_4 = get_word_embedding(
    sentences[3],
    "mouse"
)


# ==========================================
# COMPARE FINAL CONTEXTUAL REPRESENTATIONS
# ==========================================

# I compared the final Transformer representations
# of "bank" in the two different contexts.

bank_similarity = F.cosine_similarity(
    bank_1.unsqueeze(0),
    bank_2.unsqueeze(0)
).item()


# I compared the final Transformer representations
# of "mouse" in the two different contexts.

mouse_similarity = F.cosine_similarity(
    mouse_1.unsqueeze(0),
    mouse_2.unsqueeze(0)
).item()


# ==========================================
# DISPLAY BANK RESULTS
# ==========================================

print("\n==========================================")
print("BANK EXPERIMENT")
print("==========================================")

print("\nSentence 1:")
print(sentences[0])

print("Tokens:")
print(tokens_1)

print("\nInitial embedding of 'bank' - Sentence 1:")
print(bank_initial_1[:10])

print("\nFinal contextual representation of 'bank' - Sentence 1:")
print(bank_1[:10])


print("\n------------------------------------------")

print("\nSentence 2:")
print(sentences[1])

print("Tokens:")
print(tokens_2)

print("\nInitial embedding of 'bank' - Sentence 2:")
print(bank_initial_2[:10])

print("\nFinal contextual representation of 'bank' - Sentence 2:")
print(bank_2[:10])


print("\n------------------------------------------")

print("\nCosine similarity between final 'bank' representations:")
print(f"{bank_similarity:.4f}")


# ==========================================
# DISPLAY MOUSE RESULTS
# ==========================================

print("\n==========================================")
print("MOUSE EXPERIMENT")
print("==========================================")

print("\nSentence 1:")
print(sentences[2])

print("Tokens:")
print(tokens_3)

print("\nInitial embedding of 'mouse' - Sentence 1:")
print(mouse_initial_1[:10])

print("\nFinal contextual representation of 'mouse' - Sentence 1:")
print(mouse_1[:10])


print("\n------------------------------------------")

print("\nSentence 2:")
print(sentences[3])

print("Tokens:")
print(tokens_4)

print("\nInitial embedding of 'mouse' - Sentence 2:")
print(mouse_initial_2[:10])

print("\nFinal contextual representation of 'mouse' - Sentence 2:")
print(mouse_2[:10])


print("\n------------------------------------------")

print("\nCosine similarity between final 'mouse' representations:")
print(f"{mouse_similarity:.4f}")

# ==========================================
# BAT EXPERIMENT
# ==========================================

print("\n==========================================")
print("BAT EXPERIMENT")
print("==========================================")

sentence1 = "The bat flew out of the cave."
sentence2 = "He hit the ball with the bat."

print("\nSentence 1:")
print(sentence1)

print("\nSentence 2:")
print(sentence2)


# Tokenize both sentences
inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")

print("\nTokens Sentence 1:")
print(tokenizer.convert_ids_to_tokens(inputs1["input_ids"][0]))

print("\nTokens Sentence 2:")
print(tokenizer.convert_ids_to_tokens(inputs2["input_ids"][0]))


# Get hidden states
with torch.no_grad():

    outputs1 = model(
        **inputs1,
        output_hidden_states=True
    )

    outputs2 = model(
        **inputs2,
        output_hidden_states=True
    )


# Get tokens
tokens1 = tokenizer.convert_ids_to_tokens(
    inputs1["input_ids"][0]
)

tokens2 = tokenizer.convert_ids_to_tokens(
    inputs2["input_ids"][0]
)


# Find the position of "bat"
bat_index1 = tokens1.index("bat")
bat_index2 = tokens2.index("bat")


# Initial embedding
initial1 = outputs1.hidden_states[0][0, bat_index1]
initial2 = outputs2.hidden_states[0][0, bat_index2]


# Later Transformer layer
later1 = outputs1.hidden_states[-1][0, bat_index1]
later2 = outputs2.hidden_states[-1][0, bat_index2]


# Calculate similarities
initial_similarity = torch.nn.functional.cosine_similarity(
    initial1.unsqueeze(0),
    initial2.unsqueeze(0)
).item()


later_similarity = torch.nn.functional.cosine_similarity(
    later1.unsqueeze(0),
    later2.unsqueeze(0)
).item()


print("\nCosine similarity at initial embedding:")
print(f"{initial_similarity:.4f}")

print("\nCosine similarity at final Transformer layer:")
print(f"{later_similarity:.4f}")