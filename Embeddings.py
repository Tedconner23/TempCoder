import asyncio 
import glob
import os
import shutil
import tempfile
import zipfile
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from git import Repo
import numpy as np
import openai
import pandas as pd
import pdfplumber
import redis
import wget
from sklearn.metrics.pairwise import cosine_similarity
from sliding import SlidingWindowEncoder

r = redis.Redis(host='localhost', port=6379, db=0)
vector_db = VectorDatabase(redis_client=r)

class VectorDatabase:
    def __init__(self, redis_client):
        self.redis_client = r
        
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
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(keys[i], similarities[i]) for i in top_indices]
    
    def recommend(self, item_key, top_k=5):
        item_embedding = np.array(self.get_item(item_key), dtype=float)
        return self.search(item_embedding, top_k)

class Utils:
    def __init__(self, engine='text-embedding-ada-002', redis_config={'host': 'localhost', 'port': 6379, 'db': 0}, 
                 data_path='../../data/', file_name='vector_database_articles_embedded'):
        self.engine = engine
        self.redis_db = redis.StrictRedis(**redis_config)
        self.embedding_cache = {}
        self.inverted_index = defaultdict(list)
        self.data_path = data_path
        self.file_name = file_name
        self.csv_file_path = os.path.join(data_path, file_name + '.csv')
        
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))
    
    def _get_single_embedding(self, text_part, **kwargs):
        response = openai.Embedding.create(input=[text_part], engine=self.engine, **kwargs)
        return response['data'][0]['embedding']
    
    def get_embedding(self, text, **kwargs):
        """Returns the embedding for the given text by caching results."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        encoder = SlidingWindowEncoder(4096, 2048)
        encoded_text = encoder.encode(text)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._get_single_embedding, part, **kwargs) for part in encoded_text]
            embeddings = [future.result() for future in futures]
            avg_embedding = np.mean(embeddings, axis=0)
        self.embedding_cache[text] = avg_embedding
        return avg_embedding
    
    def get_relevant_memories(self, history, prompt):
        """Returns relevant memories from history based on prompt using embeddings."""
        if not history:
            return []
        prompt_embedding = self.get_embedding(prompt)
        history_embeddings = [self.get_embedding(memory) for memory in history]
        similarity_scores = self.cosine_similarity([prompt_embedding], history_embeddings)
        top_k_indices = similarity_scores.argsort()[-3:][::-1]
        return [history[i] for i in top_k_indices]
    
    def add_to_index(self, memory_id, memory_text):
        for word in memory_text.split():
            self.inverted_index[word].append(memory_id)
            
    def get_top_k_relevant_memories(self, history, prompt, k=3):
        embeddings = [self.get_embedding(string) for string in [prompt] + history if string]
        if len(embeddings) <= 1:
            return []
        similarity_scores = self.cosine_similarity([embeddings[0]], embeddings[1:])
        top_k_indices = similarity_scores.argsort()[-k:][::-1]
        return [history[i] for i in top_k_indices]
    
    def download_data(self, data_url, download_path='./'):
        zip_file_path = os.path.join(download_path, self.file_name + '.zip')
        if os.path.isfile(self.csv_file_path):
            print('File already downloaded')
        elif os.path.isfile(zip_file_path):
            print('Zip downloaded but not unzipped, unzipping now...')
        else:
            print('File not found, downloading now...')
            wget.download(data_url, out=download_path, bar=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)
        os.remove(zip_file_path)
        print(f"File downloaded to {self.data_path}")
        
    def read_data(self):
        data = pd.read_csv(self.csv_file_path)
        data['title_vector'] = data.title_vector.apply(literal_eval)
        data['content_vector'] = data.content_vector.apply(literal_eval)
        data['vector_id'] = data['vector_id'].apply(str)
        return data
    
    def save_data(self, data):
        data.to_csv(self.csv_file_path, index=False)
        print(f"Data saved to {self.csv_file_path}")
        
    def update_data(self, new_data):
        if os.path.isfile(self.csv_file_path):
            data = pd.read_csv(self.csv_file_path)
            updated_data = pd.concat([data, new_data], ignore_index=True)
            self.save_data(updated_data)
        else:
            self.save_data(new_data)
    
    def filter_data(self, column_name, value):
        data = self.read_data()
        filtered_data = data[data[column_name] == value]
        return filtered_data
    
    def filter_data_multiple_conditions(self, conditions):
        data = self.read_data()
        for (column_name, value) in conditions.items():
            data = data[data[column_name] == value]
        return data
    
    def ingest_file_content(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        allowed_extensions = ['.pdf', '.py', '.js', '.cs', '.txt', '.md']
        if file_extension not in allowed_extensions:
            raise ValueError(f"File extension '{file_extension}' is not supported. Allowed extensions are: {','.join(allowed_extensions)}")
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = pdfplumber.open(file)
                num_pages = len(pdf_reader.pages)
                content = []
                for page in range(num_pages):
                    page_text = pdf_reader.pages[page].extract_text()
                    content.append(page_text)
                return '\n'.join(content)
        else:
            with open(file_path, 'r') as file:
                return file.read()
                
    def ingest_local_repository(self, repo_path):
        content = []
        for (root, _, files) in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_content = self.ingest_file_content(file_path)
                    content.append(file_content)
                except ValueError as e:
                    print(f"Skipping file '{file_path}': {e}")
        return content
    
    async def ingest_git_repo(self, repo_url: str, file_types: list=['.cs', '.html', '.js', '.py']):
   print('Ingesting Git repository...')
   tmp_dir = tempfile.mkdtemp()
   repo = Repo.clone_from(repo_url, tmp_dir)
   
   async def process_file(file_path):
       print(f"Processing file: {file_path}")
       with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
           content = f.read()
       analysis = self.get_embedding(content)
       if analysis:
           self.redis_db.set(f"repo:{os.path.relpath(file_path, tmp_dir)}", analysis)
       else:
           print(f"No analysis data for {file_path}")
           
   tasks = []
   for file_path in glob.glob(tmp_dir + '/**', recursive=True):
       if os.path.isfile(file_path) and os.path.splitext(file_path)[-1] in file_types:
           tasks.append(asyncio.ensure_future(process_file(file_path)))
   await asyncio.gather(*tasks)
   shutil.rmtree(tmp_dir)
   print('Ingestion complete.')

	async def ingest_pdf_files(self, directory: str):
	   print('Ingesting PDF files...')
	   pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
	   
	   async def process_pdf(pdf_file):
		   print(f"Processing PDF: {pdf_file}")
		   with pdfplumber.open(pdf_file) as pdf:
			   content = ''.join(page.extract_text() for page in pdf.pages)
		   analysis = self.get_embedding(content)
		   if analysis:
			   self.redis_db.set(f"pdf:{os.path.basename(pdf_file)}", analysis)
		   else:
			   print(f"No analysis data for {pdf_file}")
			   
	   await asyncio.gather(*(process_pdf(pdf_file) for pdf_file in pdf_files))
	   print('PDF ingestion complete.')
	   
	async def get_pdf_library(self) -> str:
	   pdf_keys = [key.decode('utf-8') for key in self.redis_db.scan_iter('pdf:*')]
	   pdf_library = ''
	   for key in pdf_keys:
		   pdf_name = key.split('pdf:')[-1]
		   pdf_content = self.redis_db.get(key).decode('utf-8')
		   pdf_library += f"{pdf_name}:\n{pdf_content}\n"
	   return pdf_library

	async def print_files_in_redis_memory(self):
	   print('\nFiles in Redis memory:')
	   print(self.get_file_library())
	   print(self.get_pdf_library())

    def ingest_file(self, history_type, content):
        embedding = self.get_embedding(content)
        self.redis_db.set(f"history:{history_type}:{len(self.redis_db.keys('history:*'))}", embedding)
