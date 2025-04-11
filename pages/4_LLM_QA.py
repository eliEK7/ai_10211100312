import math
import streamlit as st
import fitz  # PyMuPDF
import time  # For measuring response time
from openai import OpenAI

## Evans Eli Kumah - 10211100312

# Function to query the Mistral API using NVIDIA's endpoint
def query_mistral_api(question, context, api_key):
    # Initialize the OpenAI client with NVIDIA's API base URL and API key
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",  # NVIDIA API URL
        api_key=api_key
    )

    # Prepare the input messages to send to the model
    messages = [
        {"role": "user", "content": f"Question: {question}\nContext: {context}"}
    ]

    # Start the timer to measure response time
    start_time = time.time()

    # Call the NVIDIA API for Mistral 7B Instruct
    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.3",  # Model you want to use
        messages=messages,
        temperature=0.2,  # Adjust creativity of the response
        top_p=0.7,  # Control response diversity
        max_tokens=1024,  # Maximum tokens for the response
        stream=True  # Stream the response to get partial content
    )
    
    # Process the streamed response
    answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            answer += chunk.choices[0].delta.content  # Append each chunk as it streams

    # Calculate the response time
    response_time = time.time() - start_time

    # Confidence will be determined by how quickly the model responds
    if response_time < 5:  # Very fast response, high confidence
        confidence_score = 1.0
    else:
        confidence_score = max(0, 1 - math.log(response_time + 1))
    
    return answer, confidence_score, response_time

st.title("Q&A: Budget Bot")

st.markdown(
    """
    **Note:** This bot only provides responses pertaining to the *Budget Statement and Economic Policy of the Government of Ghana for the 2025 Financial Year*. 
    Please ask questions related to the content of this document.
    """
)

# Specify the fixed path to dataset
pdf_path = "data/2025budget.pdf" 

# Extract text from the fixed PDF
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Extract text from the fixed PDF
pdf_text = extract_pdf_text(pdf_path)

# Split the text into chunks to avoid token limit issues
def split_text_into_chunks(text, max_tokens=3500):  # Adjust max_tokens based on limit
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        current_chunk.append(word)
        current_token_count += 1  # A rough estimate, adjust according to tokenizer

        # If the current chunk exceeds the token limit, start a new chunk
        if current_token_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add remaining words as the last chunk

    return chunks

# Split the PDF text into smaller chunks
text_chunks = split_text_into_chunks(pdf_text)

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ask a question
question = st.text_input("Ask a question:")

if question:
    # Use the first chunk as context, or you can implement logic to choose the most relevant chunk
    context = text_chunks[0]  # You can improve this by selecting more relevant chunks based on the question

    # Get API key from environment variable
    api_key = st.secrets["NVIDIA_API_KEY"]
    
    # Get the response from Mistral API
    result, confidence, response_time = query_mistral_api(question, context, api_key)
    
    # Store the user input and model response in the chat history
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", result))

    # Display the confidence and response time
    st.markdown(f"**Confidence Score**: {confidence:.2f}")
    st.markdown(f"**Response Time**: {response_time:.2f} seconds")

# Display chat history
for msg_type, msg in st.session_state.chat_history:
    if msg_type == "user":
        st.markdown(f"<div style='text-align: right; background-color: #e1f5fe; padding: 10px; border-radius: 10px; max-width: 70%; margin: 5px;'>You:<br>{msg}</div>", unsafe_allow_html=True)
    elif msg_type == "bot":
        st.markdown(f"<div style='background-color: #fff3e0; padding: 10px; border-radius: 10px; max-width: 70%; margin: 5px;'>2025 Budget Bot:<br>{msg}</div>", unsafe_allow_html=True)
