import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------------------------------------------
# TASK: Data Cleaning, BERT Embedding Generation, and FAISS Indexing
# This script "trains" and builds the system. Run this only once.
# ----------------------------------------------------------------

def build_engine():
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    data_path = 'data/tmdb_5000_movies.csv'
    processed_data_path = 'data/processed_movies.csv'
    faiss_index_path = 'data/vector_index.index'

    print("STEP 1: Loading Dataset...")
    
    if not os.path.exists(data_path):
        print(f"WARNING: '{data_path}' not found! Creating dummy data for demonstration...")
        dummy_data = {
            'original_title': ['Inception', 'The Matrix', 'Interstellar', 'The Dark Knight', 'Toy Story'],
            'overview': [
                'A thief who steals corporate secrets through the use of dream-sharing technology...',
                'A computer hacker learns from mysterious rebels about the true nature of his reality...',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival...',
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham...',
                'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him...'
            ],
            'genres': ['Action Sci-Fi', 'Action Sci-Fi', 'Adventure Sci-Fi', 'Action Crime', 'Animation Comedy']
        }
        df = pd.DataFrame(dummy_data)
    else:
        # Read the real dataset and drop missing values
        df = pd.read_csv(data_path)
        df = df[['original_title', 'overview', 'genres']].dropna()
        # Limiting to 5000 movies for performance
        df = df.head(5000) 

    print("STEP 2: Feature Engineering (Combining text features)...")
    # Combine overview and genres into a single text block for the AI to read
    df['combined_text'] = df['overview'] + " " + df['genres'].astype(str)
    
    # Save the processed data so the API can use it later
    df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to: {processed_data_path}")

    print("STEP 3: Loading Deep Learning Model (BERT)...")
    # all-MiniLM-L6-v2 is highly efficient and excellent for sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("STEP 4: Converting movies to Vectors (Embeddings)...")
    print("Please wait, this might take a minute or two...")
    texts = df['combined_text'].tolist()
    
    # Convert sentences into 384-dimensional spatial coordinates
    vectors = model.encode(texts, show_progress_bar=True)
    vectors = np.array(vectors).astype('float32') # FAISS requires float32

    print("STEP 5: Building FAISS Vector Database...")
    # Normalize vectors (Mandatory for Cosine Similarity using Inner Product)
    faiss.normalize_L2(vectors)
    
    # Create the FAISS index (Dimension: 384)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(vectors)

    # Save the database to disk
    faiss.write_index(index, faiss_index_path)
    print(f"SUCCESS! {index.ntotal} movie vectors indexed.")
    print(f"Database saved to: {faiss_index_path}")
    print("\nEngine is ready! You can now run the FastAPI server.")

if __name__ == "__main__":
    build_engine()