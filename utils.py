import asyncio, zipfile, glob, os, shutil, tempfile, zipfile, openai, pdfplumber, redis, wget, pandas as pd, numpy as np
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from git import Repo
from sklearn.metrics.pairwise import cosine_similarity
from slidingwindow import SlidingWindowEncoder
from vector import VectorDatabase

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
        """Returns relevant memories from history based
        on prompt using embeddings."""
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

    def download_and_ingest_data(self, data_url, download_path='./'):
        file_path = self.download_file(data_url, download_path)
        data = self.ingest_content(file_path)
        if isinstance(data, pd.DataFrame):
            self.save_data(data)
        else:
            print(f"Data from {file_path} was not saved because it's not a DataFrame.")

    def download_file(self, data_url, download_path='./'):
        local_filename = data_url.split('/')[-1]
        file_path = os.path.join(download_path, local_filename)
        if os.path.isfile(file_path):
            print('File already downloaded')
        else:
            print('File not found, downloading now...')
            wget.download(data_url, out=download_path, bar=True)
        return file_path

    def save_data(self, data):
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.csv_file_path, index=False)
            print(f"Data saved to {self.csv_file_path}")
        else:
            print("Data is not a DataFrame and was not saved.")

    def update_data(self, new_data):
        if os.path.isfile(self.csv_file_path):
            data = self.ingest_content(self.csv_file_path)
            if isinstance(data, pd.DataFrame) and isinstance(new_data, pd.DataFrame):
                updated_data = pd.concat([data, new_data], ignore_index=True)
                self.save_data(updated_data)
            else:
                print("One of the data is not a DataFrame and was not updated.")
        else:
            self.save_data(new_data)

    def read_data(self):
        if os.path.isfile(self.csv_file_path):
            return self.ingest_content(self.csv_file_path)
        else:
            print(f"No data found at {self.csv_file_path}")
            return None

    def filter_data(self, column_name, value):
        data = self.read_data()
        if isinstance(data, pd.DataFrame):
            filtered_data = data[data[column_name] == value]
            return filtered_data
        else:
            print("Data is not a DataFrame and was not filtered.")
            return None

    def filter_data_multiple_conditions(self, conditions):
        data = self.read_data()
        if isinstance(data, pd.DataFrame):
            for (column_name, value) in conditions.items():
                data = data[data[column_name] == value]
            return data
        else:
            print("Data is not a DataFrame and was not filtered.")
            return None

    def ingest_content(self, path):
        if os.path.isfile(path):
            return self.ingest_file_content(path)
        elif os.path.isdir(path):
            return self.ingest_directory_content(path)
        else:
            raise ValueError(f"'{path}' is neither a valid file nor a directory.")

    def ingest_file_content(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.pdf':
            return self.ingest_pdf(file_path)
        elif file_extension in ['.py', '.js', '.cs', '.txt', '.md']:
            return self.ingest_text_file(file_path)
        elif file_extension == '.csv':
            return self.ingest_csv_file(file_path)
        else:
            raise ValueError(f"File extension '{file_extension}' is not supported.")

    def ingest_directory_content(self, directory_path):
        content = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_content = self.ingest_file_content(file_path)
                    content.append(file_content)
                except ValueError as e:
                    print(f"Skipping file '{file_path}': {e}")
        return content

    def ingest_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = pdfplumber.open(file)
            num_pages = len(pdf_reader.pages)
            content = []
            for page in range(num_pages):
                page_text = pdf_reader.pages[page].extract_text()
                content.append(page_text)
            return '\n'.join(content)

    def ingest_text_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def ingest_csv_file(self, file_path):
        return pd.read_csv(file_path)
