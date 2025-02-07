import os
import re
import fitz  # PyMuPDF
import gradio as gr
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key. Please set it in the .env file.")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Load PDF document and extract text
def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text("text") for page in doc]

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the PDF inside the 'Dataset' folder
pdf_document = os.path.join(project_dir, "Dataset", "the_nestle_hr_policy_pdf_2012.pdf")

# Ensure the file exists
if not os.path.exists(pdf_document):
    raise FileNotFoundError(f"Error: The PDF file was not found at {pdf_document}")

pages = load_pdf_text(pdf_document)

documents = [Document(page_content=page_text) for page_text in pages]

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Create vector store
vectorstore = Chroma.from_documents(documents, embeddings)

# Preprocess query
def preprocess_query(query):
    query = re.sub(r'[^a-zA-Z\s]', '', query.lower())  # Remove special characters, convert to lowercase
    stopwords = {"the", "is", "in", "and", "of", "for", "to", "a"}  # Set for fast lookup
    return ' '.join(word for word in query.split() if word not in stopwords)

# Retrieve most relevant text and generate response
def chatbot_response(user_query, history=[]):
    query = preprocess_query(user_query)
    results = vectorstore.similarity_search(query, k=3)
    
    relevant_text = ' '.join([res.page_content for res in results]) if results else "I'm sorry, I couldn't find enough information related to your query."
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Build conversation history (Gradio already handles this, so don't manually append)
    messages = [{"role": "system", "content": "You are an HR assistant for Nestlé. Refer to the HR policy document to answer questions, and clearly state if there isn't enough information available."}]
    
    # Append past conversation (Gradio handles history, so we only use it here)
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Append current query
    messages.append({"role": "user", "content": user_query})

    # Generate response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    bot_reply = response.choices[0].message.content

    return bot_reply  # ✅ FIXED: No manual history appending





# Create Gradio Chatbot Interface
iface = gr.ChatInterface(
    chatbot_response,
    title="Nestlé HR Assistant",
    description="Ask any questions related to Nestlé's HR policies. This chatbot remembers context.",
)

# Launch Gradio app
if __name__ == "__main__":
    iface.launch()
