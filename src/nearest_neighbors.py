import faiss


class NearestNeighborsSearch:
    def __init__(self, embedding_dim, n_neighbors=10):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.n_neighbors = n_neighbors

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), self.n_neighbors)
        return indices[0]