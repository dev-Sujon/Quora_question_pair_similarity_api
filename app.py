import streamlit as st
import subprocess
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import time

# Main content
st.title("**Question Similarity Checker**")
st.write("Exploring Question Similarity: Techniques, Tools, and Best Practices")
# User input for questions
question1 = st.text_input("Enter the first question:")
question2 = st.text_input("Enter the second question:")

# Sidebar for selecting approach and EDA button
selected_approach = st.sidebar.selectbox("Select Approach", ["Machine Learning", "Hugging Face", "Deep Learning"])

# Function to compute similarity
def compute_similarity(model, question1, question2):
    embeddings = model.encode([question1, question2])
    similarity_score = 1 - cosine(embeddings[0], embeddings[1])
    return similarity_score

# Load the selected model
try:
    if selected_approach == "Machine Learning":
        model = SentenceTransformer("sentence-transformers/paraphrase-albert-base-v2")
        threshold = 0.85  # Define threshold for Machine Learning approach
    elif selected_approach == "Hugging Face":
        model = SentenceTransformer("sentence-transformers/stsb-mpnet-base-v2")
        threshold = 0.80  # Define threshold for Hugging Face approach
    elif selected_approach == "Deep Learning":
        model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")
        threshold = 0.50 
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Check similarity button
if st.button("Check Similarity"):
    try:
        similarity_score = compute_similarity(model, question1, question2)
        if similarity_score >= threshold:
            st.write("**Similarity Score :**",similarity_score)
            st.write("**Duplicate [1]**")
        else:
            st.write("Similarity Score :", similarity_score)
            st.write("**Not Duplicate [0]**")
    except Exception as e:
        st.error(f"**Error computing similarity:** {e}")

# Define function to run subprocess
def run_dashboard():
    try:
        subprocess.run(["streamlit", "run", "dashboard.py"])
        time.sleep(5)  # Add a delay of 5 seconds before shutting down Streamlit
    except Exception as e:
        st.error(f"Error opening dashboard: {e}")

# Move the "EDA" button below the selected approach
if st.sidebar.button("EDA", key="eda_sidebar_button"):
    run_dashboard()
