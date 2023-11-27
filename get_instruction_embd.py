import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

csv_file = "./alpaca_data.csv"
df = pd.read_csv(csv_file, usecols=['instruction', 'input', 'output'])
model = SentenceTransformer('all-MiniLM-L6-v2')

sentence_embeddings = model.encode(df['instruction'].tolist(), convert_to_tensor=True)
embeddings_list = sentence_embeddings.cpu().numpy().tolist()

# Create a DataFrame with instruction and embeddings
embeddings_df = pd.DataFrame({'instruction': df['instruction'], 'embedding': embeddings_list})

# Save the DataFrame to a CSV file
embeddings_df.to_csv("sentence_embeddings.csv", index=False)
