import faiss
import numpy as np


class NearestNeighborsSearch:
    """
    Class for simple image search using faiss 
    """

    def __init__(
            self,
            embedding_dim: int
    ):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(
            self,
            embeddings: np.ndarray
    ):
        self.index.add(embeddings)

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 6
    ):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return indices[0]
