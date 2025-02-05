# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from api import *

# Define default questions to show in sidebar
default_questions = [
    "What are the sales trends?",
    "Show me the customer demographics.", 
    "What is the revenue forecast?"
]

# Configure the Streamlit page layout
st.set_page_config(page_title="💬 Chatbot", layout="wide")

# Add custom CSS for header styling
st.markdown(
    """
    <style>
        .header-container {
            top: 55px;
            left: 10px;
            z-index: 1000;
        }
    </style>
    <div class="header-container">
        <h3>💬 Chatbot</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display default questions in sidebar
st.sidebar.title("Default Questions:")
for question in default_questions:
    st.sidebar.code(question, language=None)

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
from api import HUGGINGFACE_API_TOKEN  # Import token from external file

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HUGGINGFACE_API_TOKEN
)

# Create tabs for different business units
tabs = st.tabs(["Business Unit 1", "Business Unit 2"])

for i, tab in enumerate(tabs, start=1):
    with tab:
        # Display business unit header
        st.header(f"Business Unit {i}")
        col2 = st.columns([4])
        # Question and response section
        with col2[0]:
            col2_1, col2_2 = st.columns([4, 1])
            with col2_1:
                question = st.text_input("", label_visibility = "hidden", key=f"question{i}")
            with col2_2:
                ask_button_clicked = st.button("Ask", key=f"ask{i}")
        # Handle question submission
        if ask_button_clicked:
            if question:
                if f"messages_{i}" not in st.session_state:
                    st.session_state[f"messages_{i}"] = [{"role": "assistant", "content": "How can I help you?"}]
                st.session_state[f"messages_{i}"].append({"role": "user", "content": question})
                # Get response from HuggingFace API
                messages = st.session_state[f"messages_{i}"]
                stream = client.chat.completions.create(
                    model="HuggingFaceTB/SmolLM2-1.7B-Instruct", 
                    messages=messages, 
                    max_tokens=500,
                    stream=True
                )
                response = "".join(chunk.choices[0].delta.content for chunk in stream)
                st.session_state[f"messages_{i}"].append({"role": "assistant", "content": response})
                st.write("Response:", response)
            else:
                st.warning("Please enter a question.")
        # Handle file upload and data visualization
        with st.sidebar:
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
                        continue
                    # Display data preview and plot
                    st.write("Data Preview:")
                    st.write(df.head())
                    st.write("Plot:")
                    plt.figure()
                    df.plot()
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error processing file: {e}")
