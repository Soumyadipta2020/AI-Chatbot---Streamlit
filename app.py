# ============================================================================ #
# Import required libraries
# ============================================================================ #

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# ============================================================================ #
# Set HuggingFace API token from environment variable or local file
# ============================================================================ #

try:
    from api import HUGGINGFACE_API_TOKEN
except ImportError:
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# ============================================================================ #
# Define default questions for each business unit
# ============================================================================ #

default_questions = {
    "Business Unit 1": [
        "What are the sales trends?",
        "Show me the customer demographics.",
        "What is the revenue forecast?",
    ],
    "Business Unit 2": [
        "What are the product performance metrics?",
        "Show me the market analysis.",
        "What is the profit margin?",
    ],
}

# ============================================================================ #
# Configure the Streamlit page layout
# ============================================================================ #

st.set_page_config(page_title="💬 Chatbot", layout="wide")

# ============================================================================ #
# Add a dropdown for selecting the business unit
# ============================================================================ #

selected_business_unit = st.sidebar.selectbox(
    "Select Business Unit:", list(default_questions.keys())
)

st.sidebar.title("Default Questions:")
for question in default_questions[selected_business_unit]:
    st.sidebar.code(question, language=None)

if st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# ============================================================================ #
# Handle file upload and data visualization
# ============================================================================ #

@st.cache_data
def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file and return a DataFrame and a preview of the data.
    Currently supports CSV and Excel files.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            # If it's an image, return None to indicate non-tabular data.
            return None, None
        return df, df.head()
    except Exception as e:
        logging.error("Failed to process file", exc_info=e)
        raise e

def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file", type=["csv", "xlsx", "png", "jpg"]
    )
    df = None
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            if uploaded_file.name.endswith((".csv", ".xlsx")):
                try:
                    df, preview = process_uploaded_file(uploaded_file)
                    st.write("Data Preview:")
                    st.write(preview)
                    st.write("Plot:")
                    plt.figure()
                    df.plot()
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            else:
                # Handle image files
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    return uploaded_file, df

# Get file upload and corresponding dataframe (if any)
uploaded_file, df = handle_file_upload()

# ============================================================================ #
# Initialize OpenAI client
# ============================================================================ #

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/", 
    api_key=HUGGINGFACE_API_TOKEN
)

# ============================================================================ #
# Display business unit header
# ============================================================================ #

st.header(selected_business_unit)

# ============================================================================ #
# Initiate chatbot conversation
# ============================================================================ #

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

def handle_new_chat_message():
    prompt = st.chat_input(f"Type your question for {selected_business_unit}")
    if prompt:
        # Incorporate uploaded file context if available
        file_context = ""
        if uploaded_file is not None:
            try:
                if df is not None:
                    file_context = f"Uploaded file preview ({uploaded_file.name}):\n{df.head().to_string()}\n"
                else:
                    file_context = f"Uploaded file received: {uploaded_file.name}"
            except Exception:
                file_context = f"Uploaded file received: {uploaded_file.name}"
        full_prompt = f"{file_context}\nUser's question: {prompt}" if file_context else prompt
        # Append user’s message and display it
        st.session_state["messages"].append({"role": "user", "content": full_prompt})
        st.chat_message("user").write(full_prompt)
        # Generate response from the API
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                messages=st.session_state["messages"],
                max_tokens=2000,
            )
            msg = response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            logging.error("Chat API error", exc_info=e)
            return
        # Append assistant response and format if intermediate thinking is provided
        st.session_state["messages"].append({"role": "assistant", "content": msg})
        # Format response with think block if present
        if "<think>" in msg or "</think>" in msg:
            if "<think>" in msg and "</think>" in msg:
                think_start = msg.index("<think>") + len("<think>")
                think_end = msg.index("</think>")
                think_text = msg[think_start:think_end]
                msg = msg.replace(f"<think>{think_text}</think>", "")
            elif "</think>" in msg and "<think>" not in msg:
                think_end = msg.index("</think>")
                think_text = msg[:think_end]
                msg = msg.replace(f"{think_text}</think>", "")
            think_text = think_text.replace("\n", "</span>\n\n<span style='background-color: #01245c;'>")
            reply = (
            f"**Thinking:**\n\n<span style='background-color: #01245c;'>{think_text}</span>"
            "\n\n**Answer**\n\n" + msg
            )
            print("Message with think")
        else:
            reply = msg
            print("Message without think")
        # Format reply for final display
        reply = reply.replace("[", "$$").replace("]", "$$")
        # Display assistant response
        st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)

# Initialize chat and render previous messages
initialize_chat()

# Handle new chat user input
handle_new_chat_message()
