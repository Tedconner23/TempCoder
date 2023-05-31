import os
import asyncio
import openai
from utils import Utils
from vector import VectorDatabase
from slidingwindow import SlidingWindowEncoder

class OpenAIInteraction:
    def __init__(self, utils: Utils, vector_db: VectorDatabase):
        self.openai = openai
        self.openai.api_key = os.environ['OPENAI_API_KEY']
        self.utils = utils
        self.vector_db = vector_db

    async def get_relevant_histories(self, prompt: str):
        prompt_embedding = self.utils.get_embedding(prompt)
        relevant_histories = self.vector_db.search(prompt_embedding, top_k=5)
        context = [history for history, _ in relevant_histories] if relevant_histories else []
        return context

    async def generate_conversation(self, prompt: str, isCompletion: bool, model: str, max_tokens: int, temperature: float, persona_traits=None):
        context = await self.get_relevant_histories(prompt)
        if isCompletion:
            if persona_traits:
                prompt += f'\nYou have the following traits: {", ".join(persona_traits)}.'
            response = self.openai.Completion.create(model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        else:
            system_messages = [{'role': 'system', 'content': 'You are an AI language model.'}]
            if persona_traits:
                system_messages.append({'role': 'system', 'content': f'You have the following traits: {", ".join(persona_traits)}.'})
            full_prompt = '\n\n'.join([str(memory) for memory in context] + [f'User: {prompt}\nAI:'])
            response = self.openai.ChatCompletion.create(model=model, messages=system_messages + [{'role': 'user', 'content': full_prompt}], max_tokens=max_tokens, n=1, temperature=temperature)
        best_response = response.choices[0]['message']['content'].strip() if response.choices else ""
        return best_response

    async def analyze_input(self, input_text: str):
        input_embeddings = self.utils.get_embedding(input_text)
        relevant_histories = self.vector_db.search(input_embeddings)
        return relevant_histories

    async def run(self, user_input: str):
        try:
            relevant_histories = await self.analyze_input(user_input)
            response = await self.generate_conversation(user_input, relevant_histories)
            return response
        except Exception as e:
            return f"An error occurred. Please try again. Error: {str(e)}"
