import numpy as np
import openai
import redis
import os
import PyPDF2
from sliding import SlidingWindowEncoder
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class TextUtils:
    def __init__(self, engine='text-embedding-ada-002', redis_config={'host':'localhost', 'port':6379, 'db':0}):
        self.engine = engine
        self.redis_db = redis.StrictRedis(**redis_config)
        self.embedding_cache = {} # Cache for storing embeddings
        self.inverted_index = defaultdict(list) # Inverted index for quickly finding relevant memories
    
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))
    
    def _get_single_embedding(self, text_part, **kwargs):
        response = openai.Embedding.create(input=[text_part], engine=self.engine, **kwargs)
        return response['data'][0]['embedding']
        
    def get_embedding(self, text, **kwargs):
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # If not in cache, generate embedding
        encoder = SlidingWindowEncoder(4096, 2048)
        encoded_text = encoder.encode(text)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._get_single_embedding, part, **kwargs) for part in encoded_text]
            embeddings = [future.result() for future in futures]

        avg_embedding = np.mean(embeddings, axis=0)

        # Store in cache
        self.embedding_cache[text] = avg_embedding

        return avg_embedding
        
    def add_to_index(self, memory_id, memory_text):
        # Split the memory text into words and add the memory ID to the list for each word
        for word in memory_text.split():
            self.inverted_index[word].append(memory_id)

    def get_relevant_memories(self, history, prompt):
        if not history:
            return []
        
        # First, get a list of the memory IDs that contain at least one of the words in the prompt
        prompt_words = set(prompt.split())
        relevant_memory_ids = set()
        for word in prompt_words:
            if word in self.inverted_index:
                relevant_memory_ids.update(self.inverted_index[word])

        # Then, calculate the similarity scores for these memories and get the top k
        relevant_histories = [history[i] for i in relevant_memory_ids]
        return self.get_top_k_relevant_memories(relevant_histories, prompt, k=3)
                
    def get_top_k_relevant_memories(self, history, prompt, k=3):
        embeddings = [self.get_embedding(string) for string in [prompt] + history if string]
        if len(embeddings) <= 1:
            return []
        similarity_scores = self.cosine_similarity([embeddings[0]], embeddings[1:])
        top_k_indices = similarity_scores.argsort()[-k:][::-1]
        return [history[i] for i in top_k_indices]
       
    def ingest_file_content(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        allowed_extensions = ['.pdf', '.py', '.js', '.cs', '.txt', '.md']

        if file_extension not in allowed_extensions:
            raise ValueError(f"File extension '{file_extension}' is not supported. Allowed extensions are: {', '.join(allowed_extensions)}")

        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                content = []
                for page in range(num_pages):
                    page_text = pdf_reader.pages[page].extract_text()
                    content.append(page_text)
                return "\n".join(content)
        else:
            with open(file_path, 'r') as file:
                return file.read()
    
    def ingest_local_repository(self, repo_path):
        content = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_content = self.ingest_file_content(file_path)
                    content.append(file_content)
                except ValueError as e:
                    print(f"Skipping file '{file_path}': {e}")
        return content
    
    def ingest_file_with_embeddings(self, content):
        encoder = SlidingWindowEncoder(8000, 6000)
        encoded_content = encoder.encode(content)
        embeddings = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._get_single_embedding, part) for part in encoded_content]
            embeddings = [future.result() for future in futures]

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding
        
    async def append_to_history(self, history_type, response):
        # Here you will need to implement how you want to store your history
        # The following is just an example
        history_key = f"history:{history_type}"
        self.redis_client.lpush(history_key, response)
        
    def initialize_redis_db(self):
        self.redis_db.flushdb()

    def clear_history(self, history_type):
        history_key = f"{history_type}_history"
        self.redis_db.delete(history_key)
    
    def check_and_fix_data_type(self, history_key):
        data_type = self.redis_db.type(history_key)
        if data_type != b'list':
            print(f"Warning: Data type mismatch for key '{history_key}'. Expected 'list', found '{data_type.decode()}'. Fixing the issue.")
            self.redis_db.delete(history_key)
            self.redis_db.lpush(history_key, '')
     
