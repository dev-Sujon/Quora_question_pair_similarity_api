import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import subprocess
import torch
import warnings
warnings.filterwarnings("ignore")


matryoshka_dim = 512

# check mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# check machine learning approach
def check_similarity_machine_learning(question1, question2):
    try:
        tokenizer_hf = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        model_hf = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

        encoded_input_1 = tokenizer_hf(question1, padding=True, truncation=True, return_tensors='pt')
        encoded_input_2 = tokenizer_hf(question2, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output_1 = model_hf(**encoded_input_1)
            model_output_2 = model_hf(**encoded_input_2)

        embedding_1 = mean_pooling(model_output_1, encoded_input_1['attention_mask'])
        embedding_2 = mean_pooling(model_output_2, encoded_input_2['attention_mask'])

        embedding_1 = embedding_1.squeeze().numpy()
        embedding_2 = embedding_2.squeeze().numpy()

        similarity_score = 1 - cosine(embedding_1, embedding_2)
        threshold = 0.8
        return similarity_score, threshold
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# check Huggingface approach
def check_similarity_hugging_face(question1, question2):
    try:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        embeddings = model.encode([question1, question2])
        similarity_score = util.pytorch_cos_sim(embeddings[0:1], embeddings[1:2]).item()
        threshold = 0.75

        return similarity_score, threshold
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# check deep learning approach
def check_similarity_deep_learning(question1, question2):
    try:
        model_dl = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        sentences = [question1, question2]
        embeddings = model_dl.encode(sentences, convert_to_tensor=True)
        embeddings = embeddings[:, :matryoshka_dim]
        similarity_score = util.pytorch_cos_sim(embeddings[0:1], embeddings[1:2]).item()
        threshold = 0.9  # Adjust threshold as needed

        return similarity_score, threshold
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# call the main function
def main():
    st.set_page_config(page_title="Questions pair similarity", page_icon=":question:")

    selected_approach = st.sidebar.selectbox("Select Approach", ['Machine Learning', 'Hugging Face', 'Deep Learning'])

    st.title("Questions pair similarity")
    st.subheader(f"Selected Approach: {selected_approach}")

    question1 = st.text_input("Enter First Question")
    question2 = st.text_input("Enter Second Question")

    if st.button("Check Similarity"):
        if selected_approach == 'Machine Learning':
            similarity_score, threshold = check_similarity_machine_learning(question1, question2)
        elif selected_approach == 'Hugging Face':
            similarity_score, threshold = check_similarity_hugging_face(question1, question2)
        elif selected_approach == 'Deep Learning':
            similarity_score, threshold = check_similarity_deep_learning(question1, question2)
        else:
            st.error("Please select a valid approach.")
            return

        if similarity_score is not None:
            st.write("Similarity Score:", similarity_score)
            if similarity_score >= threshold:
                st.write("These question pairs are duplicates.")
            else:
                st.write("These question pairs are not duplicates.")

    # Move the "EDA" button below the selected approach
    if st.sidebar.button("EDA"):
        subprocess.Popen(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    main()
