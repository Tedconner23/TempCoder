from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty, ListProperty, NumericProperty
from openai_interaction import OpenAIInteraction

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

    def build(self):
        layout = BoxLayout(orientation='vertical')
        layout.padding = [20, 20, 20, 20]

        self.prompt_input = TextInput(multiline=False, font_size=24)
        layout.add_widget(self.prompt_input)

        self.history_label = Label(text=self.history, markup=True, font_size=18, size_hint_y=None, height=400)
        scrollview = ScrollView(do_scroll_x=False, do_scroll_y=True)
        scrollview.add_widget(self.history_label)
        layout.add_widget(scrollview)

        self.add_button(layout, 'Enter', self.enter_pressed)
        self.add_button(layout, 'CreativeResponse', self.creative_pressed)
        self.add_button(layout, 'CodeResponse', self.code_pressed)
        self.add_button(layout, 'Exit', self.exit_pressed)
        self.add_button(layout, 'Clear', self.clear_pressed)

        return layout

    def add_button(self, layout, text, on_press_method):
        button = Button(text=text, font_size=24)
        button.bind(on_press=on_press_method)
        layout.add_widget(button)

    async def enter_pressed(self, instance):
        prompt = self.prompt_input.text
        persona = self.selected_persona
        response = await self.interaction.generate_conversation(prompt, persona)
        self.update_history(prompt, response)

    async def creative_pressed(self, instance):
        prompt = self.prompt_input.text
        response = await self.interaction.generate_response(prompt, response_type='creative')
        self.update_history(prompt, response)

    async def code_pressed(self, instance):
        code_prompt = self.prompt_input.text
        response = self.interaction.get_code_recommendations(code_prompt)
        self.update_history(code_prompt, response)

    def clear_pressed(self, instance):
        self.prompt_input.text = ""
        self.history = ""
        self.history_label.text = self.history

    def exit_pressed(self, instance):
        App.get_running_app().stop()

    def on_stop(self):
        pass

    def update_history(self, prompt, response):
        self.history += f"\nUser: {prompt}\nAI: {response}"
        self.history_label.text = self.history
        self.prompt_input.text = ""

if __name__ == '__main__':
    interaction = OpenAIInteraction()
    OpenAIInteractionApp(interaction).run()