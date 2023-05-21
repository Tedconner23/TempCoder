import os
import asyncio
import openai
from utils import Utils
from vector import VectorDatabase
from slidingwindow import SlidingWindowEncoder

class OpenAIInteraction:
    def __init__(self):
        self.openai = openai
        self.openai.api_key = os.environ['OPENAI_API_KEY']
        self.embedding_utils = Utils()
        self.vector_db = VectorDatabase(self.embedding_utils.redis_db)
        self.encoder = SlidingWindowEncoder(4096, 2048)
        self.model_map = {'creative': 'gpt-4', 'factual': 'gpt-3.5', 'code': 'babbage', 'explanation': 'ada', 'embedding': 'embed-ada'}

    async def get_relevant_histories(self, prompt):
        prompt_embedding = self.encoder.get_embeddings(prompt)
        relevant_histories = await self.vector_db.search(prompt_embedding, top_k=5)
        context = [history for history, _ in relevant_histories]
        return context

    async def generate_response(self, prompt, context=[], response_type='creative', **kwargs):
        model = self.model_map[response_type]
        response = self.openai.Completion.create(
            model=model,
            prompt=prompt,
            context=context,
            max_tokens=8000,
            **kwargs
        )
        best_response = response.choices[0]['text'].strip()
        return best_response

    async def generate_conversation(self, prompt, persona_traits=None):
        context = await self.get_relevant_histories(prompt)
        system_messages = [{'role': 'system', 'content': 'You are an AI language model.'}]
        if persona_traits:
            system_messages.append({'role': 'system', 'content': f"You have the following traits: {', '.join(persona_traits)}."})
        full_prompt = '\n\n'.join([str(memory) for memory in context] + [f"User: {prompt}\nAI:"])
        response = self.openai.ChatCompletion.create(
            model='gpt-3.5',
            messages=system_messages + [{'role': 'user', 'content': full_prompt}],
            max_tokens=8000,
            n=1,
            stop=None,
            temperature=.7
        )
        best_response = response.choices[0]['message']['content'].strip()
        return best_response

    def get_code_recommendations(self, code_prompt, context=[]):
        response = self.openai.Codex.create(
            prompt=code_prompt + '\n\n# Here are some recommendations based on the code and context:',
            temperature=0.3,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            context=context
        )
        return response.choices[0].text.strip()

    def evaluate_text(self, text_prompt, context=[]):
        response = self.openai.Completion.create(
            model='gpt-3.5',
            prompt=text_prompt + '\n\nHuman: Here is my evaluation of the text:',
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            context=context
        )
        return response.choices[0].text.strip()

    def continue_story(self, story_context, story_prompt):
        response = self.openai.Completion.create(
            model="gpt-4",
            prompt=f"{story_context}\n\nHuman: {story_prompt}\nHuman: Here is a continuation of the story:",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        story_continuation = response.choices[0].text.strip()
        return story_continuation

    def answer_question(self, question, content):
        response = self.openai.Completion.create(
            model="gpt-4",
            prompt=f"{content}\n\nHuman: {question}\nHuman: Here is the answer to the question:",
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].text.strip()
        return answer

    def explain_response(self, explanation_prompt):
        response = self.openai.Completion.create(
            model="ada",
            prompt=explanation_prompt + '\n\nHere is an explanation for the AI response:',
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()

    async def analyze_input(self, input_text):
        input_embeddings = self.encoder.get_embeddings(input_text)
        relevant_histories = self.embedding_utils.search(input_embeddings)
        return relevant_histories

    async def run(self, user_input):
        try:
            relevant_histories = await self.analyze_input(user_input)
            response = await self.generate_response(user_input, relevant_histories)
            return response
        except Exception as e:
            return f"An error occurred. Please try again. Error: {str(e)}"
