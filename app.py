# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from api import HUGGINGFACE_API_TOKEN

# Define default questions for each business unit
default_questions = {
    "Business Unit 1": [
        "What are the sales trends?",
        "Show me the customer demographics.",
        "What is the revenue forecast?"
    ],
    "Business Unit 2": [
        "What are the product performance metrics?",
        "Show me the market analysis.",
        "What is the profit margin?"
    ]
}

# Configure the Streamlit page layout
st.set_page_config(page_title="💬 Chatbot", layout="wide")

# Add a dropdown for selecting the business unit
selected_business_unit = st.sidebar.selectbox("Select Business Unit:", list(default_questions.keys()))

# Display default questions in sidebar based on selected business unit
st.sidebar.title("Default Questions:")
for question in default_questions[selected_business_unit]:
    st.sidebar.code(question, language=None)

# Handle file upload and data visualization
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "png", "jpg"])
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

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HUGGINGFACE_API_TOKEN
)

# Display business unit header
st.header(selected_business_unit)

# Handle question submission
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input(f"Type your question for {selected_business_unit}"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        messages=st.session_state["messages"],
        max_tokens=500
    )
    msg = response.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
