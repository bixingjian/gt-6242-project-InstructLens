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
import tiktoken

st.set_page_config(page_icon="🤖", layout="wide")
st.markdown("<h2 style='text-align: center;'>InstructLens: A Toolkit for Visualizing Instructions via Aggregated Semantic and Linguistic Rules</h2>", unsafe_allow_html=True)

def get_token_count(sentence: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(sentence))
    return num_tokens

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

    # with col1:
    # Search form in the left column
    form = st.form('Embeddings')
    question = form.text_input("Enter a sentence to search for semantic similarity", 
                                value="How can we reduce air pollution?")
    num_sentences = form.number_input("Number of similar sentences to display", min_value=1, max_value=total_documents, value=3)
    btn = form.form_submit_button("Run")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    col1, col2 = st.columns([2, 3])  # Two columns for different contents

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
                ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], c='yellow', edgecolors='black', label='Query', s=20)

                ax.legend()
                ax.grid(False)
                st.pyplot(fig)

    
    with col2:
        if btn:
            with st.spinner("Searching for similar sentences..."):
                # with st.expander("See Most Similar Sentences"):
                    # Start of the fixed-height expander content
                    # st.markdown('<div class="fixed-height-expander">', unsafe_allow_html=True)
                question_embedding = model.encode(question, convert_to_tensor=True)
                saved_embeddings_df = pd.read_csv("sentence_embeddings.csv", converters={'embedding': literal_eval})
                sentence_embeddings = torch.tensor(saved_embeddings_df['embedding'].tolist())
                
                # Calculate similarities
                similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0].cpu().numpy()
                
                # Get the top similar sentence indices
                top_indices = np.argsort(similarities)[::-1][1:num_sentences+1]

                # Create a DataFrame for the similar sentences
                similar_sentences_df = pd.DataFrame({
                    'Similarity': similarities[top_indices],
                    'Instruction': df.iloc[top_indices]['instruction'],
                    'Output': df.iloc[top_indices]['output']
                })

                # Display the DataFrame
                st.write(similar_sentences_df)

                    # st.markdown('</div>', unsafe_allow_html=True)

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
    # with col2:
    #     if btn:
    #         with st.spinner("Searching for similar sentences..."):
    #             with st.expander("See Most Similar Sentences"):
    #                 # Start of the fixed-height expander content
    #                 st.markdown('<div class="fixed-height-expander">', unsafe_allow_html=True)
    #                 similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0].cpu().numpy()

    #                 # Get the top 5 most similar sentence indices
    #                 top_indices = np.argsort(similarities)[::-1][1:num_sentences+1]

    #                 # Start of the scrollable container
    #                 st.subheader("Top Similar Sentences:")
    #                 for idx in top_indices:
    #                     st.write(f"Instruction: {df.iloc[idx]['instruction']}")
    #                     st.write(f"Output: {df.iloc[idx]['output']}")
    #                     st.write(f"Similarity: {similarities[idx]:.4f}")
    #                     st.write("---------")
    #                 st.markdown('</div>', unsafe_allow_html=True)


    # single sentence word len + token
    st.subheader("Single Sentence Analysis")
    token_counts = []
    char_lengths = []
    for index, row in df.iterrows():
        combined_text = ' '.join([str(row['instruction']), str(row['input']), str(row['output'])])
        token_counts.append(get_token_count(combined_text))
        char_lengths.append(len(combined_text))

    # Create charts side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Chart 1: Token Count Distribution
    axes[0].hist(token_counts, bins=20, edgecolor='black')
    axes[0].set_xlabel('Token Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Token Counts')

    # Chart 2: Character Length Distribution
    axes[1].hist(token_counts, bins=20, edgecolor='black')
    axes[1].set_xlabel('Character Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Character Lengths')

    for ax in axes:
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)

    # Calculate averages
    avg_char_length = sum(char_lengths) / len(char_lengths)
    avg_token_count = sum(token_counts) / len(token_counts)

    # Display charts using Streamlit
    st.pyplot(fig)

    # Display average character length and token count
    st.write(f"Average Character Length: {avg_char_length:.2f}")
    st.write(f"Average Token Count: {avg_token_count:.2f}")

    st.divider()

    st.subheader("a new section")

if __name__ == "__main__":
    main()
