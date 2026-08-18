import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# PHASE 1: MANUAL VECTOR PLAYGROUND
# ==========================================

# Our manually created 2D vectors

vectors = {
    "Cat": np.array([8, 3]),
    "Dog": np.array([9, 5]),
    "Lion": np.array([7, 6]),

    "Car": np.array([1, 9]),
    "Bus": np.array([2, 8]),
    "Bicycle": np.array([3, 6]),

    "Apple": np.array([6, 2]),
    "Banana": np.array([5, 3]),
    "Pizza": np.array([4, 4])
}


# ==========================================
# PRINT THE VECTORS
# ==========================================

print("Our vectors:\n")

for name, vector in vectors.items():
    print(f"{name}: {vector}")


# ==========================================
# EUCLIDEAN DISTANCE
# ==========================================

cat_dog_distance = np.linalg.norm(
    vectors["Cat"] - vectors["Dog"]
)

cat_car_distance = np.linalg.norm(
    vectors["Cat"] - vectors["Car"]
)

print("\nEuclidean Distance:")
print(f"Cat -> Dog: {cat_dog_distance:.2f}")
print(f"Cat -> Car: {cat_car_distance:.2f}")


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


cat_dog_similarity = cosine_similarity(
    vectors["Cat"],
    vectors["Dog"]
)

cat_car_similarity = cosine_similarity(
    vectors["Cat"],
    vectors["Car"]
)

print("\nCosine Similarity:")
print(f"Cat -> Dog: {cat_dog_similarity:.3f}")
print(f"Cat -> Car: {cat_car_similarity:.3f}")


# ==========================================
# NEAREST NEIGHBOUR
# ==========================================

def find_nearest_neighbour(target_name):
    target_vector = vectors[target_name]

    distances = {}

    for name, vector in vectors.items():
        if name != target_name:
            distance = np.linalg.norm(target_vector - vector)
            distances[name] = distance

    nearest = min(distances, key=distances.get)

    return nearest, distances[nearest]


nearest_name, nearest_distance = find_nearest_neighbour("Cat")

print("\nNearest Neighbour:")
print(f"Cat's nearest neighbour is {nearest_name}")
print(f"Distance: {nearest_distance:.2f}")


# ==========================================
# 2D PLOT
# ==========================================

plt.figure(figsize=(10, 7))

for name, vector in vectors.items():
    x = vector[0]
    y = vector[1]

    plt.scatter(x, y)
    plt.text(x + 0.1, y + 0.1, name)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Manual 2D Vector Playground")

plt.grid(True)
plt.savefig("vector_plot.png")
plt.show()