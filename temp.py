import openai
import asyncio
import numpy as np
from Utils import SlidingWindowEncoder

class EmbeddingTools:
    def __init__(self, gpt_interaction):
        self.gpt_interaction = gpt_interaction

    async def cosine_similarity(self, a, b):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the cosine similarity between two vectors {a} and {b}.")

    async def euclidean_distance(self, a, b):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the Euclidean distance between two vectors {a} and {b}.")

    async def manhattan_distance(self, a, b):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the Manhattan distance between two vectors {a} and {b}.")

    async def normalize_embedding(self, embedding):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Normalize the given vector {embedding}.")

    async def get_similar_texts(self, query_embedding, text_embeddings, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} similar texts for the given query embedding {query_embedding} and text embeddings {text_embeddings}.")

    async def get_similar_texts_custom_metric(self, query_embedding, text_embeddings, metric_function, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} similar texts for the given query embedding {query_embedding} and text embeddings {text_embeddings} using the custom metric function {metric_function}.")

    async def recommend(self, query, texts, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Recommend the top {top_k} texts for the given query {query} and texts {texts}.")

    async def get_average_embedding(self, texts, model='text-davinci-002'):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the average embedding for the given texts {texts} using the model {model}.")

    async def get_nearest_neighbors(self, query_embedding, text_embeddings, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} nearest neighbors for the given query embedding {query_embedding} and text embeddings {text_embeddings}.")

    async def search(self, query, texts, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Search the top {top_k} texts for the given query {query} and texts {texts}.")

    async def search_content(self, query: str, top_k: int = 5) -> str:
        return await self.gpt_interaction.user_to_gpt_interaction(f"Search the top {top_k} content for the given query {query}.")

    async def recommend_content(self, item_key: str, top_k: int = 5) -> str:
        return await self.gpt_interaction.user_to_gpt_interaction(f"Recommend the top {top_k} content for the given item key {item_key}.")
        
    async def get_similar_texts(self, gpt_interaction, query_embedding: np.ndarray, text_embeddings: list, top_k: int = 5) -> list:
        similarities = [gpt_interaction.embeddings_tools.cosine_similarity(query_embedding, text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(similarities)[::-1]
        return [(index, similarities[index]) for index in sorted_indices[:top_k]]

    async def recommend_based_on_query(self, gpt_interaction, query: str, texts: list, top_k: int = 5) -> list:
        query_embedding = await gpt_interaction.embeddings_tools.get_ada_embeddings([query])[0]
        text_embeddings = await gpt_interaction.embeddings_tools.get_ada_embeddings(texts)
        return await self.get_similar_texts(gpt_interaction, query_embedding, text_embeddings, top_k)

    async def get_average_embedding(self, gpt_interaction, texts: list, model: str = 'text-ada-002') -> np.ndarray:
        embeddings = await gpt_interaction.embeddings_tools.get_ada_embeddings(texts, model)
        return np.mean(embeddings, axis=0)

    async def get_nearest_neighbors(self, gpt_interaction, query_embedding: np.ndarray, text_embeddings: list, top_k: int = 5) -> list:
        distances = [np.linalg.norm(query_embedding - text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(distances)
        return [(index, distances[index]) for index in sorted_indices[:top_k]]

    async def search_based_on_query(self, gpt_interaction, query: str, texts: list, top_k: int = 5) -> list:
        query_embedding = await gpt_interaction.embeddings_tools.get_ada_embeddings([query])[0]
        text_embeddings = await gpt_interaction.embeddings_tools.get_ada_embeddings(texts)
        return await self.get_nearest_neighbors(gpt_interaction, query_embedding, text_embeddings, top_k)
        
    def unique_values(self, column_name):
        data = self.read_data()
        return data[column_name].unique()

    def basic_statistics(self, column_name):
        data = self.read_data()
        return data[column_name].describe()

    def top_n_most_frequent(self, column_name, n=10):
        data = self.read_data()
        return data[column_name].value_counts().nlargest        
        
    async def process_redis_memory_context():
        keys = vector_db.get_keys_by_prefix('repo:') + vector_db.get_keys_by_prefix('pdf:')
        embeddings = [np.array(vector_db.get_item(key), dtype=float) for key in keys]
        ada_embeddings = await get_ada_embeddings([key for key in keys])
        return dict(zip(keys, ada_embeddings))

    async def get_memory_keywords(redis_memory_context, ada_embeddings, threshold=0.8):
        memory_keywords = []
        for key, value in redis_memory_context.items():
            similarity = cosine_similarity(ada_embeddings, value)
            if similarity >= threshold:
                memory_keywords.append(key)
        return memory_keywords

    async def keywordizer(planning, input_text, redis_memory_context):
        tasks_to_execute = []
        ada_embeddings = await get_ada_embeddings(input_text)
        memory_keywords = await get_memory_keywords(redis_embeddings)

        for task in planning.get_tasks():
            all_keywords = task.keywords + memory_keywords
            if any(re.search(f"\\b{keyword}\\b", input_text, re.IGNORECASE) for keyword in all_keywords):
                tasks_to_execute.append(task)
            else:
                inferred_keywords = await get_similar_keywords(ada_embeddings, task.keywords)
                if inferred_keywords:
                    tasks_to_execute.append(task)

        return tasks_to_execute

import os, wget, zipfile, numpy as np, pandas as pd
from ast import literal_eval

class Utils:
    def __init__(self, data_path='../../data/', file_name='vector_database_articles_embedded'):
        self.data_path = data_path
        self.file_name = file_name
        self.csv_file_path = os.path.join(data_path, file_name + '.csv')

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
        for column_name, value in conditions.items():
            data = data[data[column_name] == value]
        return data
        
    def ensure_output_directory_exists():
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

    def write_response_to_file(response, filename):
        with open(os.path.join(OUTPUT_DIRECTORY, filename), 'w', encoding='utf-8') as file:
            file.write(response)

class IngestFiles:
    async def ingest_git_repo(repo_url: str, file_types: list = ['.cs', '.html', '.js', '.py']):
        print("Ingesting Git repository...")
        tmp_dir = tempfile.mkdtemp()
        repo = Repo.clone_from(repo_url, tmp_dir)

        async def process_file(file_path):
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            analysis = await get_ada_embeddings(content)
            if analysis:
                vector_db.add_item(f"repo:{os.path.relpath(file_path, tmp_dir)}", analysis)
            else:
                print(f"No analysis data for {file_path}")

        tasks = [asyncio.ensure_future(process_file(file_path)) for file_path in glob.glob(tmp_dir + '/**', recursive=True) if os.path.isfile(file_path) and os.path.splitext(file_path)[-1] in file_types]
        await asyncio.gather(*tasks)
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        print("Ingestion complete.")

    async def ingest_pdf_files(directory: str):
        print("Ingesting PDF files...")
        pdf_files = glob.glob(os.path.join(directory, '*.pdf'))

    async def process_pdf(pdf_file):
        print(f"Processing PDF: {pdf_file}")
        with pdfplumber.open(pdf_file) as pdf:
            content = ''.join(page.extract_text() for page in pdf.pages)
            analysis = await get_ada_embeddings(content)
            if analysis:
                vector_db.add_item(f"pdf:{os.path.basename(pdf_file)}", analysis)
            else:
                print(f"No analysis data for {pdf_file}")
                
        await asyncio.gather(*(process_pdf(pdf_file) for pdf_file in pdf_files))
        print("PDF ingestion complete.")

    async def get_pdf_library() -> str:
        pdf_keys = vector_db.get_keys_by_prefix('pdf:')
        pdf_library = ''
        for key in pdf_keys:
            pdf_name = key.split('pdf:')[-1]
            pdf_content = r.get(key).decode('utf-8')
            pdf_library += f"{pdf_name}:\n{pdf_content}\n"
        return pdf_library

    async def print_files_in_redis_memory():
        print('\nFiles in Redis memory:')
        keys = vector_db.get_keys_by_prefix('pdf:')
        for key in keys:
            pdf_name = key.split('pdf:')[-1]
            print(f"- {pdf_name}")


import json

CONVERSATION_HISTORY_KEY = 'conversation_history'

class ConversationHistory:
    def __init__(self):
        self.interactions = []
        self.documents = []
        self.queries = []

    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def add_document(self, document):
        self.documents.append(document)

    def add_query(self, query):
        self.queries.append(query)

    def get_interactions(self):
        return self.interactions

    def get_documents(self):
        return self.documents

    def get_queries(self):
        return self.queries

    def save_to_redis(self, r):
        r.set(CONVERSATION_HISTORY_KEY, self.serialize())

    def load_from_redis(self, r):
        history_data = r.get(CONVERSATION_HISTORY_KEY)
        if history_data is not None:
            self.deserialize(history_data.decode('utf-8'))

    def serialize(self):
        return json.dumps({'interactions': self.interactions,
                           'documents': self.documents,
                           'queries': self.queries})

    def deserialize(self, data):
        history_data = json.loads(data)
        self.interactions = history_data['interactions']
        self.documents = history_data['documents']
        self.queries = history_data['queries']

    def clear_history(self, r):
        self.interactions = []
        self.documents = []
        self.queries = []
        r.delete(CONVERSATION_HISTORY_KEY)


class TaskContext:
    def __init__(self, task_id, related_files=None, code_snippets=None, tools_and_methods=None):
        self.task_id = task_id
        self.related_files = related_files if related_files is not None else []
        self.code_snippets = code_snippets if code_snippets is not None else []
        self.tools_and_methods = tools_and_methods if tools_and_methods is not None else {}

    def add_related_file(self, file):
        self.related_files.append(file)

    def add_code_snippet(self, snippet):
        self.code_snippets.append(snippet)

    def add_tool_and_method(self, tool, method):
        if tool not in self.tools_and_methods:
            self.tools_and_methods[tool] = []
        self.tools_and_methods[tool].append(method)


class PlanningContext(TaskContext):  # Inheriting from TaskContext to avoid repetition
    def __init__(self, goal_context, *args, **kwargs):  # Using *args and **kwargs for flexibility
        super().__init__(*args, **kwargs)  # Call parent constructor
        self.goal_context = goal_context
        self.related_files = related_files if related_files is not None else []
        self.relevant_knowledge = relevant_knowledge if relevant_knowledge is not None else {}
        self.tasks = tasks if tasks is not None else []

    def add_related_file(self, file):
        self.related_files.append(file)

    def add_relevant_knowledge(self, knowledge):
        self.relevant_knowledge[knowledge] = True

    def add_task(self, task):
        task.set_plan(self)
        self.tasks.append(task)
        
class Task:
    def __init__(self,
                 goal,
                 task_context=None,
                 specific_instructions=None,
                 methods=None,
                 dependencies=None,
                 context=None):
        self.goal = goal
        self.task_context = task_context if task_context is not None else TaskContext('')
        self.specific_instructions = specific_instructions if specific_instructions is not None else []
        self.methods = methods if methods is not None else []
        self.dependencies = dependencies if dependencies is not None else []
        self.context = context if context is not None else {}

    def set_task_context(self, task_context):
        self.task_context = task_context

    def get_task_context(self):
        return self.task_context

    def update_goal(self, new_goal):
        self.goal = new_goal

    def add_related_file(self, file):
        self.related_files.append(file)

    def add_code_snippet(self, snippet):
        self.code_snippets.append(snippet)

    def add_specific_instruction(self, instruction):
        self.specific_instructions.append(instruction)

    def add_method(self, method):
        self.methods.append(method)

    def refine_goal(self, new_goal):
        self.methods = new_goal

    def update_related_files(self, files):
        self.related_files = files

    def update_code_snippets(self, snippets):
        self.code_snippets = snippets

    def update_specific_instructions(self, instructions):
        self.specific_instructions = instructions

    def present_and_review_methods(self, methods):
        self.methods = methods

    def update_task_context(self, key, value):
        self.task_context[key] = value


class Planning:
    def __init__(self,
                 tasks=None,
                 iterations=1,
                 context=None,
                 planning_context=None):
        self.tasks = tasks if tasks is not None else []
        self.iterations = iterations
        self.context = context if context is not None else {}
        self.planning_context = planning_context if planning_context is not None else PlanningContext('')

    def set_planning_context(self, planning_context):
        planning_context

    def get_planning_context(self):
        return self.planning_context

    def add_task(self, task):
        task.set_plan(self)
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks

    def refine_task(self, task, goal=None, related_files=None, code_snippets=None, specific_instructions=None):
        if goal:
            task.refine_goal(goal)
        if related_files:
            task.update_related_files(related_files)
        if code_snippets:
            task.update_code_snippets(code_snippets)
        if specific_instructions:
            task.update_specific_instructions(specific_instructions)

    def set_iterations(self, iterations):
        self.iterations = iterations

    def update_context(self, context):
        self.context = context

    def execute_finalized_plan(self):
        pass

    def get_sorted_tasks(self):
        return sorted(self.tasks, key=lambda t: (-t.priority, t.goal))

    def get_task_dependencies(self, task):
        return [t for t in self.tasks if t.goal in task.dependencies]

    def set_context(self, context):
        self.context = context

    def add_snippet(self, snippet):
        self.snippets.append(snippet)

    def add_external_link(self, link):
        self.external_links.append(link)

    def set_repo(self, repo):
        self.repo = repo

    def add_highlighted_file(self, file):
        self.highlighted_files.append(file)

    def execute_tasks(self):
        for _ in range(self.iterations):
            sorted_tasks = self.get_sorted_tasks()
            for task in sorted_tasks:
                dependencies = self.get_task_dependencies(task)
                for dependency in dependencies:
                    print(f"Executing dependency: {dependency.goal}")
                    dependency.method()
                print(f"Executing task: {task.goal}")
                task.method()

    def update_planning_context(self, key, value):
        self.planning_context[key] = value
        
import os
import os
import glob
import openai
import shutil
import stat
import tempfile
import pdfplumber
from git import Repo
import redis
import asyncio
from datastructures import CONVERSATION_HISTORY_KEY
from gpt import gpt_interaction, SlidingWindowEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        keys = self.get_keys_by_prefix('repo:') + self.get_keys_by_prefix('pdf:')
        embeddings = [np.array(self.get_item(key), dtype=float) for key in keys]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(keys[i], similarities[i]) for i in top_indices]

    def recommend(self, item_key, top_k=5):
        item_embedding = np.array(self.get_item(item_key), dtype=float)
        return self.search(item_embedding, top_k)
import asyncio
import openai
from Config import *
from EmbeddingTools import EmbeddingTools
from Utils import *
from RedisDB import VectorDatabase
from ChatHistory import ConversationHistory, CONVERSATION_HISTORY_KEY
from PlanTask import TaskContext, PlanningContext, Task, Planning
import numpy as np

class GPTInteraction:
    def __init__(self, model_data):
        self.model_data = model_data
        self.embedding_tools = EmbeddingTools(api_key)
        self.utils = Utils()
        self.vector_db = VectorDB()
    
    async def get_embeddings(self, text, model_name):
        model = self.model_data[model_name]
        loop = asyncio.get_event_loop()
        window_size = model['window_size']
        step_size = model['step_size']
        sliding_window_encoder = SlidingWindowEncoder(window_size, step_size)
        windows = sliding_window_encoder.encode(text)

        async def get_embeddings(window):
            response = await loop.run_in_executor(None, lambda: openai.Completion.create(engine=model_name, prompt=window, max_tokens=1, n=1, stop=None, temperature=.5))
            return response.choices[0].text.strip()

        embeddings = await asyncio.gather(*(get_embeddings(window) for window in windows))
        return embeddings

    async def user_to_gpt_interaction(self, prompt, model_name):
        model = self.model_data[model_name]
        max_tokens = self.calculate_max_tokens(model)
        RESERVED_TOKENS = 50
        CHUNK_SIZE = max_tokens - RESERVED_TOKENS

        system_message = {'role': 'system', 'content': f"You are an {','.join(ASSISTANT_TRAITS)} assistant. Use Ada agents and other tools for GPT-3.5-turbo to aid in interactions and responses."}
        user_message = {'role': 'user', 'content': prompt}
        messages = [system_message, user_message]

        response = await self._execute_chat_completion(model_name, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']

        return response_text.strip()

    async def gpt_to_gpt_chat_interaction(self, prompt, model, max_tokens, toAgent, toAgentHistory, fromAgent, fromAgentHistory):
        messages = [
            {'role': 'user', 'content': f"You are asking a question of, or commanding, {toAgent} to resolve: {prompt}"},
            {'role': 'system', 'content': f"You are interacting with {toAgent}. The prompt to resolve is: {prompt}", 'traits': '', 'name': 'fromAgent', 'history': 'fromAgentHistory', 'interlocutor': 'toAgent', 'tools_and_methods': 'Available tools to call: tools_and_methods(reduced by context)'},
            {'role': 'assistant', 'content': f"You are interacting with {toAgent}. The prompt to resolve is: {prompt}", 'traits': '', 'name': 'toAgent', 'history': 'toAgentHistory', 'interlocutor': 'fromAgentHistory', 'tools_and_methods': 'Available tools to call: tools_and_methods(reduced by context)'}
        ]

        response = await self._execute_chat_completion(model, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']
        return response_text.strip()

    async def complete_prompt(self, prompt, model, max_tokens, message):
        messages = [{'role': 'user', 'content': f"As a basic prompt completion bot, you are with a variation of the message: {message}. Keep meaning but the language does not have to be exact."}]
        response = await self._execute_chat_completion(model, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']
        return response_text.strip()

    async def respond_to_prompt(self, prompt, model, max_tokens, history, string_in, task_in, personality, name, responsibility):
        m1 = [{'role': 'user', 'content': f"Generate prompt instructions for text-davinci-003 where the instructions are: You are {name} and have personality: {personality} with task: {task_in} as context."}]
        response1 = await self._execute_chat_completion(model, m1, max_tokens)
        generated_prompt = response1['choices'][0]['message']['content']

        m2 = [{'role': 'user', 'content': f"You are replying to the user after post analysis. Your response prompt is: {generated_prompt}"}]
        response2 = await self._execute_chat_completion(model, m2, max_tokens)
        response_text = response2['choices'][0]['message']['content']
        return response_text.strip()

    async def gpt_to_user_interaction(self, prompt, model, max_tokens):
        RESERVED_TOKENS = 50
        CHUNK_SIZE = max_tokens - RESERVED_TOKENS

        system_message = {'role': 'system', 'content': f"You are an {','.join(ASSISTANT_TRAITS)} assistant. Use Ada agents and other tools for GPT-3.5-turbo to aid in interactions and responses."}
        user_message = {'role': 'user', 'content': prompt}
        messages = [system_message, user_message]

        response = await self._execute_chat_completion(model, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']
        return response_text.strip()

    async def _execute_chat_completion(self, model_name, messages, max_tokens):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: openai.ChatCompletion.create(model=model_name, messages=messages, max_tokens=max_tokens, n=1, stop=None, temperature=.7))
        return response

    @staticmethod
    def calculate_max_tokens(model: dict, remaining_tokens: int) -> int:
        model_token_limit = int(remaining_tokens * .8)
        RESERVED_TOKENS = 50
        max_tokens = model_token_limit - RESERVED_TOKENS
        HARD_LIMIT = min(model['token_limits'], 1000)
        max_tokens = min(max_tokens, HARD_LIMIT)
        return max_tokens

class SlidingWindowEncoder:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def encode(self, input_data):
        if len(input_data) <= self.window_size:
            return [input_data]
        windows = []
        start = 0
        end = self.window_size
        while end <= len(input_data):
            windows.append(input_data[start:end])
            start += self.step_size
            end += self.step_size
        if start < len(input_data):
            windows.append(input_data[start:])
        return windows

    def decode(self, encoded_data):
        decoded_data = []
        for window in encoded_data:
            decoded_data.extend(window)
        return decoded_data    
from PlanTask import TaskContext, PlanningContext, Task, Planning
from ChatHistory import ConversationHistory
from EmbeddingTools import EmbeddingTools
from GPT import GPTInteraction
import multiprocessing

embedding_tools_methods=[{'method_name':'search_content','input':('query','top_k'),'possible_output':'list of search results'},{'method_name':'recommend_content','input':('item_key','top_k'),'possible_output':'list of recommended items'},{'method_name':'get_similar_texts','input':('query_embedding','text_embeddings','top_k'),'possible_output':'list of similar texts'},{'method_name':'recommend_based_on_query','input':('query','texts','top_k'),'possible_output':'list of recommended texts'},{'method_name':'get_average_embedding','input':('texts','model'),'possible_output':'average embedding vector'},{'method_name':'get_nearest_neighbors','input':('query_embedding','text_embeddings','top_k'),'possible_output':'list of nearest neighbors'},{'method_name':'search_based_on_query','input':('query','texts','top_k'),'possible_output':'list of search results'},{'method_name':'unique_values','input':('column_name',),'possible_output':'list of unique values'},{'method_name':'basic_statistics','input':('column_name',),'possible_output':'dictionary containing basic statistics'},{'method_name':'top_n_most_frequent','input':('column_name','n'),'possible_output':'list of top n most frequent items'}]

class ExecutiveAgent:
    def __init__(self, model_data, agent_id):
        self.agent_id = agent_id
        self.name = f"Executive{agent_id}"
        self.model_data = model_data
        self.processes = []

    def spawn_worker(self, worker_class, *args):
        process = multiprocessing.Process(target=worker_class, args=args)
        process.start()
        self.processes.append(process)

    def subagent__receive(self, worker_class, *args):
        # Receive messages from subagents
        # Process the messages and decide the next course of action
        pass

    def subagent__send(self, worker_class, *args):
        # Send messages to subagents
        # Coordinate the actions of subagents
        pass

    def exec_decide(self, worker_class, *args):
        # Make decisions based on subagent input and the current state of the system
        # This method should be called periodically to maintain the overall control of the system
        pass

    def exec_run_method(self, worker_class, *args):
        # Execute specific methods of subagents
        # This method can be used to directly control the actions of subagents
        pass

    def exec_run_command(self, worker_class, *args):
        # Execute specific commands for subagents
        # This method can be used to send commands to subagents, which they will execute
        pass

    def terminate_all_agents(self):
        for process in self.processes:
            process.terminate()
        print("All agents terminated by ExecutiveAgent")

    def terminate_individual_agent(self, agent_name):
        for process in self.processes:
            if process._args[0].name == agent_name:
                process.terminate()
                print(f"{agent_name} terminated by ExecutiveAgent")
                return
        print(f"Agent {agent_name} not found")
        
class ExecutiveSubAgent:
    def __init__(self, model_data, agent_id):
        self.agent_id = agent_id
        self.name = f"Executive{agent_id}"
        self.model_data = model_data
        self.processes = []

    def track_n_clean(self, request):
        # Update and clean process list of running agents
        # Analyze completion time vs estimated task time
        # Alert exec to long-running tasks or agents
        pass

    def receive_user_interrupt(self, request):
        # System input from user
        # Used to pause, redirect, change, etc, as per user's wishes, directly to the highest hierarchy
        # Input is put through intent analysis, then sent to exec
        pass

    def receive_request(self, request):
        # Receive the request of agents and analyze complexity
        # If complex beyond fast models, send to exec
        pass

    def analyze_request(self, request):
        # Analyze the request and decide action
        # Use fast models unless complexity > threshold
        pass
        
class LeadAgent:
    def __init__(self, model, level, traits, tools, agent_id, parent_agent=None):
        self.model = model
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"Lead{agent_id}"
        self.parent_agent = parent_agent

    def spawn_worker(self, worker_class, *args):
        process = multiprocessing.Process(target=worker_class, args=args)
        process.start()
        self.parent_agent.processes.append(process)

    def terminate_code_ada_agents(self):
        for process in self.parent_agent.processes:
            if process._args[0] in [CodeAgent, AdaAgent] and process._args[0].parent_agent == self:
                process.terminate()
                print(f"{process._args[0].name} terminated by Lead Agent")

class BrowsingAgent:
    def __init__(self, model, level, traits, tools, agent_id, parent_agent=None):
        self.model = model
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"Codex{agent_id}"
        self.parent_agent = parent_agent

    def execute_task(self):
        super().execute_task()
        gpt_interaction_script(self.model, self.task_context, self.planning_context)

class CodeAgent:
    def __init__(self, model, level, traits, tools, agent_id, parent_agent=None):
        self.model = model
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"Codex{agent_id}"
        self.parent_agent = parent_agent

    def execute_task(self):
        super().execute_task()
        gpt_interaction_script(self.model, self.task_context, self.planning_context)        

class EmbedAgent:
    def __init__(self, model, level, traits, tools, agent_id, parent_agent=None):
        self.model = model
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"Ada{agent_id}"
        self.parent_agent = parent_agent
        self.embedding_tools = EmbeddingTools('')
        self.embedding_tools_methods=[{'method_name':'search_content','input':('query','top_k'),'possible_output':'list of search results'},{'method_name':'recommend_content','input':('item_key','top_k'),'possible_output':'list of recommended items'},{'method_name':'get_similar_texts','input':('query_embedding','text_embeddings','top_k'),'possible_output':'list of similar texts'},{'method_name':'recommend_based_on_query','input':('query','texts','top_k'),'possible_output':'list of recommended texts'},{'method_name':'get_average_embedding','input':('texts','model'),'possible_output':'average embedding vector'},{'method_name':'get_nearest_neighbors','input':('query_embedding','text_embeddings','top_k'),'possible_output':'list of nearest neighbors'},{'method_name':'search_based_on_query','input':('query','texts','top_k'),'possible_output':'list of search results'},{'method_name':'unique_values','input':('column_name',),'possible_output':'list of unique values'},{'method_name':'basic_statistics','input':('column_name',),'possible_output':'dictionary containing basic statistics'},{'method_name':'top_n_most_frequent','input':('column_name','n'),'possible_output':'list of top n most frequent items'}]

    def search_content(self, query, top_k=5):
        return self.embedding_tools.search_based_on_query(query, self.planning_context.related_files, top_k)

    def recommend_content(self, item_key, top_k=5):
        return self.embedding_tools.recommend_based_on_query(item_key, self.planning_context.related_files, top_k)

    def call_tool_method(self, method_name, *args, **kwargs):
        if hasattr(self.embedding_tools, method_name):
            method = getattr(self.embedding_tools, method_name)
            return method(*args, **kwargs)
        else:
            raise AttributeError(f"EmbeddingTools does not have a method named '{method_name}'")

class MemoryAgent:
    def __init__(self, model, level, traits, tools, agent_id, parent_agent=None):
        self.model = model
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"Ada{agent_id}"
        self.parent_agent = parent_agent

    def search_conv_history(self, query, top_k=5):        
    def search_redis(self, query, top_k=5):        

    def recommend_content(self, item_key, top_k=5):
        return self.embedding_tools.recommend_based_on_query(item_key, self.planning_context.related_files, top_k) 

class Agent:
    def __init__(self, model, role, level, traits, tools, agent_id, process, parent_agent=None):
        self.model = model
        self.role = role
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"{role}{agent_id}"
        self.process = process
        self.parent_agent = parent_agent
        self.task_context = TaskContext('')
        self.planning_context = PlanningContext('')
        self.embedding_tools = EmbeddingTools('')

    def notify_admin(self):
        print(f"Admin notified by {self.name}")

    def engage_user(self):
        print(f"User engaged by {self.name}")

    def self_assessment(self):
        print(f"Self-assessment performed by {self.name}")

    def execute_task(self):
        print(f"Task executed by {self.name}")

    def halt_execution(self):
        print(f"Execution halted by {self.name}")

    def terminate_agent_self(self):
        print(f"Agent {self.name} self terminated")


if __name__ == '__main__':
    agent_parameters = 'model', 'level', 'traits', 'tools', 'agent_id'
    executive_agent = ExecutiveAgent(model_data, '001')
    lead_agent = LeadAgent(*agent_parameters, parent_agent=executive_agent)
    code_agent = CodeAgent(*agent_parameters, parent_agent=lead_agent)
    ada_agent = AdaAgent(*agent_parameters, parent_agent=lead_agent)
    executive_agent.spawn_worker(LeadAgent, *agent_parameters, executive_agent)
    lead_agent.spawn_worker(CodeAgent, *agent_parameters, lead_agent)
    lead_agent.spawn_worker(AdaAgent, *agent_parameters, lead_agent)
