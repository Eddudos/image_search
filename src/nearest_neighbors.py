import faiss


class NearestNeighborsSearch:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=6):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return indices[0]