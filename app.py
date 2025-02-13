# ============================================================================ #
# Import required libraries
# ============================================================================ #

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import os

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

# ============================================================================ #
# Display default questions in sidebar based on selected business unit
# ============================================================================ #

st.sidebar.title("Default Questions:")
for question in default_questions[selected_business_unit]:
    st.sidebar.code(question, language=None)

# ============================================================================ #
# Handle file upload and data visualization
# ============================================================================ #

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a file", type=["csv", "xlsx", "png", "jpg"]
    )
    if uploaded_file is not None:
        try:
            # Read and process different file types
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            # Display data preview and plot
            st.write("Data Preview:")
            st.write(df.head())
            st.write("Plot:")
            plt.figure()
            df.plot()
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Error processing file: {e}")

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

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# ============================================================================ #
# Display chat messages
# ============================================================================ #

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ============================================================================ #
# Handle user input and generate response
# ============================================================================ #

if prompt := st.chat_input(f"Type your question for {selected_business_unit}"):
    # Incorporate uploaded file context if available
    file_context = ""
    if uploaded_file is not None:
        try:
            # If the uploaded file was processed into a DataFrame (CSV or Excel), use its preview
            file_context = f"Uploaded file preview ({uploaded_file.name}):\n{df.head().to_string()}\n"
        except Exception:
            # For other file types (like images), include basic file info
            file_context = f"Uploaded file received: {uploaded_file.name}"
    # Combine file context with the user's prompt if context exists
    full_prompt = f"{file_context}\nUser's question: {prompt}" if file_context else prompt
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": full_prompt})
    st.chat_message("user").write(full_prompt)
    # Generate response from OpenAI API
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        messages=st.session_state["messages"],
        max_tokens=2000,
    )
    msg = response.choices[0].message.content
    # Add assistant response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": msg})
    # Extract and format the intermediate thinking text if available
    if "<think>" in msg and "</think>" in msg:
        think_start = msg.index("<think>") + len("<think>")
        think_end = msg.index("</think>")
        think_text = msg[think_start:think_end]
        msg = msg.replace(f"<think>{think_text}</think>", "")
        # Format the thinking text with background color and heading
        think_text = think_text.replace("\n", "</span>\n\n<span style='background-color: blue;'>")
        reply = (
            f"**Thinking:**\n\n<span style='background-color: #01245c;'>{think_text}</span>"
            + "\n\n**Answer**\n\n"
            + msg
        )
    else:
        reply = msg
    # Render mathematical equations properly (if any)
    reply = reply.replace("[", "$$").replace("]", "$$")
    st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)
