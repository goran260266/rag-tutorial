"""Plot functions for the vector embeddings slides."""

import matplotlib.pyplot as plt


def plot_2d_vectors(a, b, c, d):
    vectors = {"a": a[0], "b": b[0], "c": c[0], "d": d[0]}
    colors = {"a": "red", "b": "black", "c": "darkblue", "d": "green"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, v in vectors.items():
        ax.annotate(
            "",
            xy=v,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colors[name], lw=2),
        )
        ax.text(v[0] + 0.15, v[1] + 0.15, name, fontsize=14, color=colors[name])
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-4, 4.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title("2D-Vektoren / 2D Vectors")
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(similarities, texts):
    plt.figure(figsize=(8, 7))
    plt.imshow(similarities, cmap="viridis", aspect="auto")
    plt.colorbar(label="Ähnlichkeit / Similarity")
    plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))])
    plt.yticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))])

    for i in range(len(texts)):
        for j in range(len(texts)):
            plt.text(
                j,
                i,
                f"{similarities[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if similarities[i, j] < 0.5 else "black",
            )

    plt.title("Semantische Ähnlichkeit / Semantic Similarity")
    plt.tight_layout()
    plt.show()
