import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import openai
from langchain.docstore.document import Document
import fitz
from frontend import *
import os

response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an HR assistant for Nestlé."},
        {"role": "user", "content": "Can you help me understand the leave policy?"}
    ]
)

# Print the chatbot's response
print(response['choices'][0]['message']['content'])

# Open the PDF file
pdf_document = "/Users/trevoralpert/Desktop/Dataset/the_nestle_hr_policy_pdf_2012.pdf"
doc = fitz.open(pdf_document)

# Extract text from each page
pages = []
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)  # load page by index
    page_text = page.get_text("text")  # extract text
    pages.append(page_text)

# Print the text from the first page
print(pages[0])

# Pass the API key directly when creating the embeddings object
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key="sk-oXvebtphtymWsKI60Nwe7-FyyjBO8p8lrxBlA4_VcrT3BlbkFJ5daHjcOnARH50tWGu_yVxdwC71Qh7woujnlriIIOMA"
)


# Convert each page into a Document format required by Chroma
documents = [Document(page_content=page_text) for page_text in pages]

# Create a vector store using Chroma and store the embedded text
vectorstore = Chroma.from_documents(documents, embeddings)

# Now, the documents are stored in Chroma and can be queried based on similarity

# Define the query (e.g., asking about the leave policy)
query = "What is Nestlé's mission?"

# Perform a similarity search on the vector store
results = vectorstore.similarity_search(query)

# Print the most relevant chunk of text from the HR policy document
if results:
    print("Relevant text found:")
    print(results[0].page_content)
else:
    print("No relevant text found.")


response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an HR assistant for Nestlé."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": results[0].page_content}
    ]
)

# Print the chatbot's response
print(response['choices'][0]['message']['content'])


import re

def preprocess_query(query):
    # Convert query to lowercase
    query = query.lower()
    
    # Remove special characters and numbers
    query = re.sub(r'[^a-z\s]', '', query)
    
    # Optionally, remove stopwords (you can expand this list)
    stopwords = ['the', 'is', 'in', 'and', 'of', 'for', 'to', 'a']
    query_words = query.split()
    query = ' '.join([word for word in query_words if word not in stopwords])
    
    return query

# Test with a sample query
query = preprocess_query("What is the company's mission?")
print(query)  # Output: "what leave policy employees"

# Retrieve the top 3 most relevant chunks
results = vectorstore.similarity_search(query, k=3)

# Combine the top results into one response
if results:
    relevant_text = ' '.join([result.page_content for result in results])
    print("Relevant text found:")
    print(relevant_text)
else:
    # Handle ambiguous or no results found
    print("I'm sorry, I couldn't find enough information related to your query.")

response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an HR assistant for Nestlé. Refer to the HR policy document to answer questions, and clearly state if there isn't enough information available."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": relevant_text if results else "I'm sorry, I couldn't find enough information related to your query."}
    ]
)

# Print the chatbot's response
print(response['choices'][0]['message']['content'])


# Ensure you have the API key and necessary imports
openai.api_key = "sk-oXvebtphtymWsKI60Nwe7-FyyjBO8p8lrxBlA4_VcrT3BlbkFJ5daHjcOnARH50tWGu_yVxdwC71Qh7woujnlriIIOMA"

# Function to handle chatbot interaction
def chatbot_response(user_query):
    # Preprocess the query and perform similarity search
    query = preprocess_query(user_query)
    
    # Retrieve the top 3 most relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    
    if results:
        # Combine the top results into one response
        relevant_text = ' '.join([result.page_content for result in results])
    else:
        relevant_text = "I'm sorry, I couldn't find enough information related to your query."
    
    # Generate the final response from the chatbot
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an HR assistant for Nestlé. Refer to the HR policy document to answer questions, and clearly state if there isn't enough information available."},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": relevant_text}
        ]
    )
    
    return response['choices'][0]['message']['content']

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot_response,  # The function that gets called with the input
    inputs="text",  # Input is a text box
    outputs="text",  # Output is displayed as text
    title="Nestlé HR",
    description="Ask any questions related to Nestlé's HR policies."
)

# Launch the Gradio interface
iface.launch()
