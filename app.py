# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from api import HUGGINGFACE_API_TOKEN

# Define default questions to show in sidebar
default_questions = [
    "What are the sales trends?",
    "Show me the customer demographics.", 
    "What is the revenue forecast?"
]

# Configure the Streamlit page layout
st.set_page_config(page_title="💬 Chatbot", layout="wide")

# # Add custom CSS for header styling
# st.markdown(
#     """
#     <style>
#         .header-container {
#             top: 55px;
#             left: 10px;
#             z-index: 1000;
#         }
#     </style>
#     <div class="header-container">
#         <h3>💬 Chatbot</h3>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# Display default questions in sidebar
st.sidebar.title("Default Questions:")
for question in default_questions:
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

# Create tabs for different business units
tabs = st.tabs(["Business Unit 1", "Business Unit 2"])

for i, tab in enumerate(tabs, start=1):
    with tab:
        # Display business unit header
        st.header(f"Business Unit {i}")
        # Handle question submission
        if f"messages_{i}" not in st.session_state:
            st.session_state[f"messages_{i}"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state[f"messages_{i}"]:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input(f"Type your question for Business Unit {i}"):
            st.session_state[f"messages_{i}"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                messages=st.session_state[f"messages_{i}"], 
                max_tokens=500
            )
            msg = response.choices[0].message.content
            st.session_state[f"messages_{i}"].append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
            
