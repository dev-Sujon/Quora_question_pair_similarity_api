import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the Streamlit app
def main():
    st.title("Quora Question Pair Similarity")
    
    # User input for two questions
    question1 = st.text_input("Enter the first question:")
    question2 = st.text_input("Enter the second question:")
    
    if st.button("Check Similarity"):
        # Encode the input questions
        embeddings = model.encode([question1, question2])
        
        # Calculate the cosine similarity between the two embeddings
        similarity_score = util.pytorch_cos_sim(embeddings[0:1], embeddings[1:2])[0][0].item()
        
        # Display the similarity score
        st.write("Similarity Score:", similarity_score)
        if similarity_score >= 0.5:  # Corrected the condition
            st.write("Duplicate[1]")
        else:
            st.write("Not Duplicate[0]")

if __name__ == "__main__":
    main()
