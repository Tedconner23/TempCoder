import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import redis

class VectorDatabase:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def add_item(self, key: str, value: list):
        value_str = ','.join(value)
        self.redis_client.set(key, value_str)

    def get_item(self, key: str) -> list:
        value_str = self.redis_client.get(key)
        if value_str is not None:
            return value_str.decode('utf-8').split(',')
        return None

    def get_keys_by_prefix(self, prefix):
        return [key.decode('utf-8') for key in self.redis_client.scan_iter(f"{prefix}*")]

    def search(self, query_embedding, top_k=5):
        keys = self.get_keys_by_prefix('history:')
        embeddings = [np.array(self.get_item(key), dtype=float) for key in keys]
        if not embeddings:
            return []
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(keys[i], similarities[i]) for i in top_indices]

    def recommend(self, item_key, top_k=5):
        item_embedding = np.array(self.get_item(item_key), dtype=float)
        return self.search(item_embedding, top_k)
