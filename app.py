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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

st.set_page_config(page_icon="🤖", layout="wide")
st.markdown("<h2 style='text-align: center;'>InstructLens: A Toolkit for Visualizing Instructions via Aggregated Semantic and Linguistic Rules</h2>", unsafe_allow_html=True)


def get_token_count(sentence: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(sentence))
    return num_tokens

# Preprocessing function
def preprocess_text(text):
    # Check if the text is NaN or None
    if pd.isna(text):
        return ""

    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    
    return " ".join(stemmed_tokens)

def perform_topic_modeling(processed_texts, num_topics=5):
    texts = [text.split() for text in processed_texts]
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=4)
    return topics

def extract_keywords(data, top_n=10):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=None, min_df=2, stop_words='english', use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(data)
    feature_array = np.array(vectorizer.get_feature_names_out())

    # Sum tfidf scores for each feature across all documents
    scores_sum = np.sum(tfidf_matrix.toarray(), axis=0)
    
    # Get the indices of the top n scores
    top_indices = np.argsort(scores_sum)[-top_n:]
    
    # Get the corresponding feature names and their scores
    top_features = feature_array[top_indices]
    top_scores = scores_sum[top_indices]

    # Create a DataFrame with the features and scores
    keywords_score_table = pd.DataFrame({'Keyword': top_features, 'Score': top_scores})
    
    # Sort by score in descending order
    keywords_score_table = keywords_score_table.sort_values(by='Score', ascending=False)
    
    return keywords_score_table

# Function to generate a word cloud
def generate_wordcloud(keywords_df):
    # Convert DataFrame to dictionary
    word_scores = keywords_df.set_index('Keyword')['Score'].to_dict()
    
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    # Generate the word cloud using frequencies
    wordcloud.generate_from_frequencies(word_scores)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())  # Use Streamlit's function to display the plot

def main():
    # Load csv file
    csv_file = "./alpaca_data.csv"
    df = pd.read_csv(csv_file, usecols=['instruction', 'input', 'output'])

    # Display the total number of documents
    st.subheader("Basic Information about the Dataset")
    total_documents = len(df)
    st.write(f"Total number of documents: **{total_documents:,}**")
    
    # Display the average length of documents
    average_document_length = df['instruction'].apply(lambda x: len(x.split())).mean()
    st.write(f"Average length of instructions (in words): **{average_document_length:.2f}**")
    
    # Display the missing values in each column
    missing_values = df.isnull().sum()
    st.write("Missing values in each column: ")
    token_counts = df['instruction'].apply(get_token_count)
    st.table(missing_values)
    
    # Display the average number of tokens per instruction
    average_token_count = token_counts.mean()
    st.write(f"Average number of tokens per instruction: **{average_token_count:.2f}**")
    
    # Function to calculate the number of words
    def word_count(text):
        if pd.isna(text):
            return 0
        return len(text.split())
    # Calculate statistics for 'instruction' and 'output'
    instruction_length = df['instruction'].apply(word_count)
    output_length = df['output'].apply(word_count)

    instruction_mean = instruction_length.mean()
    instruction_std = instruction_length.std()
    output_mean = output_length.mean()
    output_std = output_length.std()

    # Display the statistics
    st.subheader("Statistical Information about the Dataset")
    st.write(f"Mean length of instructions (in words): **{instruction_mean:.2f}** ± **{instruction_std:.2f}**")
    st.write(f"Mean length of outputs (in words): **{output_mean:.2f}** ± **{output_std:.2f}**")
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

                st.write(similar_sentences_df)


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

    nltk.download('punkt')
    nltk.download('stopwords')
    # Preprocess the dataset
    df['instruction_processed'] = df['instruction'].apply(preprocess_text)
    df['output_processed'] = df['output'].apply(preprocess_text)
    
    # Topic Modeling
    st.subheader("Topic Modeling")
    topics = perform_topic_modeling(df['instruction_processed'])
    for topic in topics:
        st.write(topic)
# Extract and display output keywords as a table
    st.subheader("Output Keywords")
    output_keywords_df = extract_keywords(df['output_processed'], top_n=10)
    st.table(output_keywords_df)

    # Visualization: Word Clouds
    st.subheader("Instruction Keywords Word Cloud")
    instruction_keywords_df = extract_keywords(df['instruction_processed'], top_n=10)
    generate_wordcloud(instruction_keywords_df)

    st.subheader("Output Keywords Word Cloud")
    generate_wordcloud(output_keywords_df)

    st.subheader("Knowledge Analysis")
    data = {
    "Topic": [
        "sentence rewrite", "given input", "list make", 
        "following classify", "generate questions", "create poem", 
        "explain concept", "write poem", "words 100", 
        "word synonym", "article summarize", "example use", 
        "identify type", "using rewrite", "provide examples", 
        "story short", "suggest new", "numbers calculate", 
        "learning machine", "text edit"
    ],
    "Frequency": [
        7.48 + 1.62, 5.28 + 0.43, 4.31 + 0.74, 
        4.60 + 0.74, 4.86 + 0.35, 4.98 + 0.40, 
        3.90 + 1.12, 4.10 + 0.41, 3.90 + 0.28, 
        3.88 + 0.34, 2.81 + 2.75, 4.54 + 0.79, 
        3.91 + 0.89, 3.98 + 1.03, 4.71 + 0.71, 
        3.35 + 0.88, 1.96 + 1.10, 2.65 + 1.39, 
        1.90 + 1.52, 4.09 + 0.53
    ]
}

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Update the topics to verb+noun format
    df["Topic"] = df["Topic"].apply(lambda x: ' '.join(x.split()[::-1]))

    # Sort the DataFrame by frequency in descending order
    df_sorted = df.sort_values(by="Frequency", ascending=False)

    st.dataframe(df_sorted)

if __name__ == "__main__":
    main()
