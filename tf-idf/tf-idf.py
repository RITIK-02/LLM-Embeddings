from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

with open("doc.txt", "r") as file:
    corpus = file.read()

N = len(corpus)
print(f"Corpus has {len(corpus.split())} words.")
print(f"Corpus has {N} characters.")
N //= 10

documents = [corpus[i:i+N] for i in range(0, len(corpus), N)]
documents = documents[:-1]

vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(documents)
words = vectorizer.get_feature_names_out()

print(f"Words count: {len(words)} e.g.: {words[-10:]}")
print(f"Embeddings shape: {embeddings.shape}")

print(embeddings)

word_vectors = embeddings.T.toarray()

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(word_vectors)

df = pd.DataFrame(reduced_embeddings, columns=["X", "Y"])
df["Words"] = words

fig = px.scatter(df, x="X", y="Y", text="Words", title="PCA of TF-IDF Embeddings")
fig.update_traces(textposition="top center", marker=dict(size=5, color="purple"))
fig.show()

tsne = TSNE(n_components=2, perplexity=10, random_state=42)
reduced_embeddings2 = tsne.fit_transform(word_vectors)

df2 = pd.DataFrame(reduced_embeddings2, columns=["X", "Y"])
df2["Words"] = words

fig2 = px.scatter(df2, x="X", y="Y", text="Words", title="t-SNE of TF-IDF Embeddings")
fig2.update_traces(textposition="top center", marker=dict(size=5, color="purple"))
fig2.show()
