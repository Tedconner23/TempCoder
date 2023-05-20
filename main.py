import os
import time
import sys
import asyncio
from colorama import Fore, Style, init
import openai
from sliding import SlidingWindowEncoder
from embeddings import Utils
from rich.console import Console
from rich.theme import Theme
from rich.progress import Progress

custom_theme = Theme({
    'info': 'dim cyan',
    'error': 'bold red',
    'success': 'bold green',
    'prompt': 'yellow',
    'completion': 'bold magenta'
})
console = Console(theme=custom_theme)
progress = Progress(console=console)

class OpenAIInteraction:
    def __init__(self):
        init(autoreset=True)
        self.openai = openai
        self.openai.api_key = os.environ['OPENAI_API_KEY']
        self.embedding_utils = Utils(redis_config={'host': 'localhost', 'port': 6379, 'db': 0})
        self.model_map = {
            'creative': 'davinci',
            'factual': 'curie',
            'code': 'babbage'
        }
        self.encoder = SlidingWindowEncoder(4096, 2048)
        
    def get_code_recommendations(self, code_prompt):
        response = openai.Codex.create(
            prompt=code_prompt + '\n\n# Here are some recommendations based on the code:',
            temperature=0.3,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()  
        
    def evaluate_text(self, text_prompt):
        response = openai.Completion.create(
            model="davinci",
            prompt=text_prompt + '\n\nHuman: Here is my evaluation of the text: ',
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
        
    def continue_story(self, story_context, story_prompt):
        response = openai.Completion.create(
            model="davinci",
            prompt=f"{story_context}\n\nHuman: {story_prompt}\nHuman: Here is a continuation of the story: ",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        story_continuation = response.choices[0].text.strip()
        return story_continuation
        
    def answer_question(self, question, content):
        response = openai.Completion.create(
            model="davinci",
            prompt=f"{content}\n\nHuman: {question}\nHuman: Here is the answer to the question: ",
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].text.strip()
        return answer
        
    def explain_response(self, explanation_prompt):
        response = openai.Completion.create(
            model="ada",
            prompt=explanation_prompt + '\n\nHere is an explanation for the AI response: ',
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
        
    def print_message(self, print_message):
        response = openai.Completion.create(
            model="babbage", 
            prompt=f"{print_message}\n\nHere is the message printed by the AI: ",
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()

    def download_file(self, url):
        local_filename = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
        return local_filename
        
    async def analyze_input(self, input_text, context=[]):
        input_embeddings = self.encoder.get_embeddings(input_text)
        relevant_histories = self.embedding_utils.search(input_embeddings)
        return relevant_histories  
        
    async def generate_response(self, prompt, relevant_histories, response_type='creative', context=[], **kwargs):
        model = self.model_map[response_type]
        if persona_traits:
            system_messages.append({'role': 'system', 'content': f"You have the following traits: {', '.join(persona_traits)}."})
        response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, top_p=top_p, context=context, **kwargs)
        best_response = response.choices[0]['text'].strip()
        return best_response
        
    async def generate_response(self, prompt, relevant_histories, persona_traits=None):
        system_messages = [{'role': 'system', 'content': 'You are an AI language model.'}]
        if persona_traits:
            system_messages.append({'role': 'system', 'content': f"You have the following traits: {', '.join(persona_traits)}."})
        full_prompt = '\n\n'.join([str(memory) for memory in relevant_histories] + [f"User: {prompt}\nAI:"])
        response = self.openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=system_messages + [{'role': 'user', 'content': full_prompt}], max_tokens=1600, n=1, stop=None, temperature=.7)
        best_response = response.choices[0]['message']['content'].strip()
        return best_response

    async def run(self):
        console.print('Initializing assistant...', style='info')
        time.sleep(2)
        welcome_prompt = await self.generate_response('Welcome to the AI Assistant. How can I help you today?', [])
        console.print(welcome_prompt, style='success')

        context = []
        
        while True:
            try:
                input_text = console.input("[prompt]\nEnter your next prompt: ")
                if input_text.lower() == 'exit':
                    console.print('Thank you for using the AI Assistant. Goodbye!', style='info')
                    break
                elif input_text.lower().startswith('code '):
                    code_prompt = input_text.split('code ', 1)[1]
                    response = self.get_code_recommendations(code_prompt)
                    console.print(response, style='success')
                elif input_text.lower().startswith('story '):
                    story_prompt = input_text.split('story ', 1)[1]
                    story_context = '\n\n'.join(self.embedding_utils.get_history('story'))
                    story_continuation = self.continue_story(story_context, story_prompt)
                    console.print(story_continuation, style='success') 
                elif input_text.lower().startswith('eval '):
                    text_prompt = input_text.split('eval ', 1)[1]
                    evaluation = self.evaluate_text(text_prompt)
                    console.print(evaluation, style='success')
                elif input_text.lower().startswith('question '):
                    question = input_text.split('question ', 1)[1]
                    content = '\n\n'.join(self.embedding_utils.get_history('content'))
                    answer = self.answer_question(question, content)
                    console.print(answer, style='success')
                elif input_text.lower().startswith('explain '):
                    explanation_prompt = input_text.split('explain ', 1)[1] 
                    explanation = self.explain_response(explanation_prompt)
                    console.print(explanation, style='success')

                elif input_text.lower().startswith('print '):
                    print_message = input_text.split('print ', 1)[1]
                    print_response = self.print_message(print_message)
                    console.print(print_response, style='success') 
                elif input_text.lower().startswith('download '):
                    url = input_text.split('download ', 1)[1]
                    self.download_file(url)
                    console.print(f"File downloaded from {url}.", style='success')
                else:
                    console.print('Analyzing input...', style='info')
                    relevant_histories = await self.analyze_input(input_text, context)
                    console.print('Generating AI response...', style='info')
                    with progress:
                        task = progress.add_task('[completion]Generating...', total=100)  
                        response = await self.generate_response(input_text, relevant_histories, context=context)  
                        progress.update(task, completed=100)
                    console.print('\nFull Response:\n', response, style='success')
                    context.append(response)  
            except KeyboardInterrupt:
                console.print('\nExiting due to KeyboardInterrupt', style='error')
                break
            except Exception as e:
                console.print(f"An unexpected error occurred: {str(e)}", style='error')

if __name__ == '__main__':
    interaction = OpenAIInteraction()
