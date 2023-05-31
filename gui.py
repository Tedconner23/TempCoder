import threading, redis, asyncio
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label  
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty, ListProperty, NumericProperty
from kivy.clock import Clock
from utils import Utils 
from vector import VectorDatabase
from openai_interaction import OpenAIInteraction

redis_config = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}
redis_client = redis.StrictRedis(**redis_config) 

class OpenAIInteractionApp(App):
    history = StringProperty()
    model = StringProperty('davinci')
    traits = ListProperty()
    temperature = NumericProperty(0.7)
    top_p = NumericProperty(1)
    max_tokens = NumericProperty(100)

    def __init__(self, interaction):
        super().__init__()
        self.interaction = interaction
        self.max_tokens = 2048
        self.temperature = 1
        
    def build(self):
        layout = BoxLayout(orientation='vertical')
        layout.padding = [20, 20, 20, 20]

        self.prompt_input = TextInput(hint_text='Enter your prompt here', multiline=False, font_size=24)
        layout.add_widget(self.prompt_input)

        self.traits_input = TextInput(hint_text='Enter persona traits here', multiline=False, font_size=24)
        layout.add_widget(self.traits_input)

        self.history_label = Label(text=self.history, markup=True, font_size=18, size_hint_y=None, height=400)
        scrollview = ScrollView(do_scroll_x=False, do_scroll_y=True)
        scrollview.add_widget(self.history_label)
        layout.add_widget(scrollview)

        self.add_button(layout, 'Enter', self.enter_pressed)
        self.add_button(layout, 'Exit', self.exit_pressed)
        self.add_button(layout, 'Clear', self.clear_pressed)

        return layout

    def add_button(self, layout, text, on_press_method):
        button = Button(text=text, font_size=24)
        button.bind(on_press=on_press_method)
        layout.add_widget(button)

    def enter_pressed(self, instance):
        threading.Thread(target=self.async_enter_pressed).start()

    def async_enter_pressed(self):
        prompt = self.prompt_input.text
        traits = self.traits_input.text.split(',')
        model = self.model_input.text
        max_tokens = int(self.tokens_input.text)
        temperature = float(self.temperature_input.text)
        response = asyncio.run(self.interaction.generate_conversation(prompt, False, model, max_tokens, temperature, persona_traits=traits))
        print(response);
        Clock.schedule_once(lambda dt: self.update_history(prompt, response), 0) 


    def exit_pressed(self, instance):
        App.get_running_app().stop()
        
    def clear_pressed(self, instance):
        self.prompt_input.text = ""
        
    def on_stop(self):
        pass

    def update_history(self, prompt, response):
        self.history += f"User: {prompt} AI: {response}"
        self.history_label.text = self.history
        self.prompt_input.text = prompt

if __name__ == '__main__':
    interaction = OpenAIInteraction(Utils(redis_client), VectorDatabase(redis_client)) 
    OpenAIInteractionApp(interaction).run()
