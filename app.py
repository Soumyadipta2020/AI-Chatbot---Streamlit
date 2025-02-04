import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import os

# Import API File
from api.R import *

# Define the default questions
default_questions = [
    "What are the sales trends?",
    "Show me the customer demographics.",
    "What is the revenue forecast?"
]

# Streamlit UI
def main():
    st.set_page_config(page_title="ChatGPT-Style App", layout="wide")
    if "asked" not in st.session_state:
        st.session_state.asked = False
    if not st.session_state.asked:
        st.markdown("<h1 style='text-align: top;'>ChatGPT-Style App</h1>", unsafe_allow_html=True)
    st.sidebar.title("Default Questions:")
    st.sidebar.write("\n".join(default_questions))
    tabs = st.tabs(["Business Unit 1", "Business Unit 2"])
    for i, tab in enumerate(tabs, start=1):
        with tab:
            st.header(f"Business Unit {i}")
            col1, col2 = st.columns([1, 3])
            with col1:
                uploaded_file = st.file_uploader(label="Upload your file", type=["csv", "xlsx", "png", "jpg"], key=f"file{i}")
            with col2:
                col2_1, col2_2 = st.columns([4, 1])
                with col2_1:
                    question = st.text_input("Ask a question", key=f"question{i}")
                with col2_2:
                    if st.button("Ask", key=f"ask{i}"):
                        if question:
                            response = openai.Completion.create(
                                engine="text-davinci-003",
                                prompt=question,
                                max_tokens=150
                            )
                            st.write("Response:", response.choices[0].text.strip())
                        else:
                            st.warning("Please enter a question.")
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                    else:
                        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                        continue
                    st.write("Data Preview:")
                    st.write(df.head())
                    st.write("Plot:")
                    plt.figure()
                    df.plot()
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
