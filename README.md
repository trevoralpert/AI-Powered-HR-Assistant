# Nestlé HR Assistant Chatbot

## Overview
This project is an AI-powered HR assistant that helps users inquire about Nestlé's HR policies. The chatbot is built using **LangChain**, **OpenAI GPT-4o**, and **Gradio**, with document retrieval powered by **ChromaDB**.

## Features
- **Conversational Memory**: The chatbot maintains the context of the conversation.
- **PDF Document Retrieval**: Queries are answered based on information extracted from the HR policy document.
- **Dynamic File Path Handling**: The chatbot automatically locates the HR policy document stored in the `Dataset/` folder.
- **Secure API Key Handling**: The OpenAI API key is stored in a `.env` file.

## Technologies Used
- **Python 3.12**
- **OpenAI GPT-4o**
- **LangChain & ChromaDB**
- **Gradio**
- **PyMuPDF (for PDF text extraction)**
- **Dotenv (for environment variables)**

## Installation

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/trevoralpert/AI-Powered-HR-Assistant.git
cd AI-Powered-HR-Assistant
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv .venv
source .venv/bin/activate  # For Mac/Linux
.venv\Scripts\activate     # For Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up the API Key
Create a `.env` file in the root directory and add:
```plaintext
OPENAI_API_KEY=your-openai-api-key
```

### 5️⃣ Run the Chatbot
```sh
python app2.py
```
The chatbot will launch in your browser at `http://127.0.0.1:7860`.

## Usage
1. Start the chatbot.
2. Ask HR-related questions.
3. The chatbot retrieves the most relevant information from the HR policy document.

## Contributing
Pull requests are welcome. If you’d like to contribute, please fork the repository and submit a PR.

## License
This project is licensed under the MIT License. See the **LICENSE** file for details.
