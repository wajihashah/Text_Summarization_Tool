import streamlit as st
from transformers import pipeline

# Load the BART model for text summarization from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit app layout
st.title("Text Summarization Tool")
st.write("Automatically summarizes long articles or documents into concise summaries using the BART model.")

# Input field for the user to enter or paste the text
input_text = st.text_area("Enter the text you want to summarize:", height=300)

# Check if there is input text
if input_text:
    # Display the original text
    st.subheader("Original Text:")
    st.write(input_text)

    # Generate the summary using the pre-trained BART model
    summary = summarizer(input_text, max_length=200, min_length=50, do_sample=False)

    # Display the summarized text
    st.subheader("Summary:")
    st.write(summary[0]['summary_text'])

# Option to clear the input
if st.button("Clear Text"):
    st.text_area("Enter the text you want to summarize:", value="")
