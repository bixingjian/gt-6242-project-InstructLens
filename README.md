# InstructLens
A Toolkit for Visualizing Instructions via Aggregated Semantic and Linguistic Rules

## Description
This package comprises a toolkit called "InstructLens" for visualizing instructions using aggregated semantic and linguistic rules. It employs various libraries such as Streamlit, Pandas, NLTK, Gensim, and others to process and analyze a dataset stored in a CSV file ("alpaca_data.csv"). The toolkit provides functionalities like data summarization (counting documents, average length, missing values), statistical analysis (mean lengths, distributions), and semantic similarity search based on sentence embeddings. It also includes text preprocessing methods like tokenization, stop words removal, stemming, and topic modeling using Latent Dirichlet Allocation (LDA). Additionally, it generates word clouds and identifies key phrases for both instructions and outputs, facilitating knowledge analysis by presenting a frequency-based summary of different topics gleaned from the dataset. The toolkit offers a streamlined interface through Streamlit for users to interactively explore and understand the dataset's textual content and semantic relationships.

## Installation
Install all dependencies `pip install -r requirements.txt`
- openai
- streamlit
- pandas
- numpy
- nomic
- matplotlib
- plotly
- scipy
- scikit-learn
- python-dotenv
- sentence_transformers
- tiktoken

## Execution
1. Run `streamlit run app.py`
2. Open a web browser to the provided localhost URL
3. Interact with the visualizations and use the semantic search functionality on the webpage
