import os
import openai
from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_icon="🤖", layout="wide")
st.markdown("<h2 style='text-align: center;'>InstructLens: A Toolkit for Visualizing Instructions via Aggregated Semantic and Linguistic Rules</h2>", unsafe_allow_html=True)

def main():
    # Load csv file
    csv_file = "./alpaca_data.csv"
    df = pd.read_csv(csv_file, usecols=['instruction', 'input', 'output'])

    # Display the total number of documents
    st.subheader("Basic Information about the Dataset")
    total_documents = len(df)
    st.write(f"Total number of documents: **{total_documents:,}**")
    st.divider()

    st.subheader("Data:")
    st.write(df)
    st.divider()

    # Display most similar 5 sentences
    st.subheader("Search similarity")
    col1, col2 = st.columns([3, 2])  # Two columns for different contents
    with col1:
        # Search form in the left column
        form = st.form('Embeddings')
        question = form.text_input("Enter a sentence to search for semantic similarity", 
                                   value="How can we reduce air pollution?")
        num_sentences = form.number_input("Number of similar sentences to display", min_value=1, max_value=total_documents, value=3)
        btn = form.form_submit_button("Run")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if btn:
        with col1: 
            with st.spinner("generating t-SNE plot..."):
                # Combine all embeddings for t-SNE (question embedding + sentence embeddings)
                # Compute embedding for the input question
                question_embedding = model.encode(question, convert_to_tensor=True)
                # Load precomputed sentence embeddings
                saved_embeddings_df = pd.read_csv("sentence_embeddings.csv", converters={'embedding': literal_eval})
                sentence_embeddings = torch.tensor(saved_embeddings_df['embedding'].tolist())
                all_embeddings = np.vstack([question_embedding.cpu().numpy(), sentence_embeddings.cpu().numpy()])

                # Perform t-SNE
                tsne = TSNE(n_components=2, random_state=0)
                embeddings_2d = tsne.fit_transform(all_embeddings)

                # Visualization
                marker_size = 1.5
                fig, ax = plt.subplots()
                colors = ['red'] + ['blue'] * len(sentence_embeddings)  # Red for query, blue for sentences
                ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=marker_size)

                # Highlight the question point
                ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], c='green', edgecolors='black', label='Query', s=10)

                ax.legend()
                ax.grid(False)
                st.pyplot(fig)

        # t-SNE plot in the right column
    expander_css = """
    <style>
    .fixed-height-expander {
        max-height: 650px;
        overflow-y: auto;
    }
    </style>
    """
    st.markdown(expander_css, unsafe_allow_html=True)

    # t-SNE plot and similar sentences in different columns
    with col2:
        if btn:
            with st.spinner("Searching for similar sentences..."):
                with st.expander("See Most Similar Sentences"):
                    # Start of the fixed-height expander content
                    st.markdown('<div class="fixed-height-expander">', unsafe_allow_html=True)
                    similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0].cpu().numpy()

                    # Get the top 5 most similar sentence indices
                    top_indices = np.argsort(similarities)[::-1][1:num_sentences+1]

                    # Start of the scrollable container
                    st.subheader("Top Similar Sentences:")
                    for idx in top_indices:
                        st.write(f"Instruction: {df.iloc[idx]['instruction']}")
                        st.write(f"Output: {df.iloc[idx]['output']}")
                        st.write(f"Similarity: {similarities[idx]:.4f}")
                        st.write("---------")
                    st.markdown('</div>', unsafe_allow_html=True)
        # with col2:
        #     with st.spinner("Searching for similar sentences..."):
        #         with st.expander("See Most Similar Sentences"):
        #         # Compute cosine similarities
        #             similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0].cpu().numpy()

        #             # Get the top 5 most similar sentence indices
        #             top_indices = np.argsort(similarities)[::-1][1:num_sentences+1]

        #             # Start of the scrollable container
        #             st.subheader("Top Similar Sentences:")
        #             for idx in top_indices:
        #                 st.write(f"Instruction: {df.iloc[idx]['instruction']}")
        #                 st.write(f"Output: {df.iloc[idx]['output']}")
        #                 st.write(f"Similarity: {similarities[idx]:.4f}")
        #                 st.write("---------")

    st.subheader("a new section")

if __name__ == "__main__":
    main()
