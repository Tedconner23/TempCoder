## AI Assistant CLI

The AI Assistant CLI is a powerful and versatile command-line interface that allows you to interact with an AI language model. With support for various history types, persona traits, and file ingestion, the AI Assistant CLI provides a seamless and efficient way to obtain answers, suggestions, and assistance from an AI model in a conversational manner.

### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [History Types](#history-types)
4. [Persona Traits](#persona-traits)
5. [File Ingestion](#file-ingestion)
6. [Commands](#commands)

### Installation

To install the AI Assistant CLI, follow these steps:

1. Clone the repository or download the source code.
2. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

3. Run the CLI using the following command:

```bash
python main.py <history_types>
```

Replace `<history_types>` with a comma-separated list of history types, such as `conversation,story,coding`.

### Usage

To use the AI Assistant CLI, launch the interface by running:

```bash
python main.py <history_types>
```

Once the CLI is running, you can interact with the AI by entering your prompts. The AI will respond with relevant information based on the conversation history, persona traits, and any ingested files.

### History Types

The AI Assistant CLI supports the following history types:

1. `conversation`: Includes conversation history, such as questions and answers.
2. `story`: Includes narrative history, such as stories, scenarios, and context.
3. `coding`: Includes coding history, such as code snippets, programming concepts, and technical explanations.

### Persona Traits

The AI Assistant CLI allows you to define persona traits for the AI model. These traits help guide the AI's behavior and responses during the conversation. Some example traits include:

- AllCodeGen
- UI/UXDesignExpert
- AllContentGen
- AIExpert
- C#CodingExpert
- CodeGuru
- UnityWhiz
- UX/UIAce
- CodeFocused
- OnPoint
- OptimizationAce

### File Ingestion

The AI Assistant CLI supports ingesting files to provide additional context and information to the AI model. Supported file formats include:

- PDF (.pdf)
- Python (.py)
- JavaScript (.js)
- C# (.cs)
- Text (.txt)
- Markdown (.md)

To ingest a file, use the `ingest` command followed by the file path.

### Commands

The AI Assistant CLI supports the following commands:

- `exit`: Exits the CLI.
- `clear`: Clears a specified history type (conversation, story, coding).
- `ingest`: Ingests a file and adds its content and embeddings to the conversation history.

To execute a command, simply enter the command name as a prompt during the conversation.

Happy chatting!
