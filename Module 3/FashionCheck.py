import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Read the cleaned CSV file
df = pd.read_csv("cleaned_fashion_data.csv")

# Define features to keep for similarity analysis
features = ['Outfit', 'Color', 'Priority', 'Wardrobe', 'Shopping_Frequency', 
            'Influence', 'Experimentation', 'Footwear', 'Activity', 'Comfort', 
            'Preference']

# Drop all other columns to avoid errors
df = df[['Timestamp'] + features].copy()  # Create a copy to avoid slice issues

# Data Cleaning
def clean_data(df):
    # Drop rows with any missing data in features
    df = df.dropna(subset=features)
    
    # Standardize text data using .loc to avoid warnings
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Timestamp':
            df.loc[:, col] = df[col].str.lower().str.strip()
    
    return df

# Apply cleaning
df = clean_data(df)

# One-Hot Encoding (updated for newer scikit-learn versions)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[features])

# Calculate Cosine Similarity
similarity_matrix = cosine_similarity(encoded_features)

# Query Selection (ensure timestamps exist in dataset)
query_timestamps = [
    '21/01/2025 23:13:08',  # Casual, Neutral, Comfort-Focused Female
    '29/01/2025 23:16:19',  # Sporty, Neutral, Functionality-Focused Male
    '14/01/2025 19:33:18'   # Chic, Pastels, Aesthetics-Focused Female
]

querys = [
    "Casual, Neutral, Comfort-Focused Female",
    "Sporty, Neutral, Functionality-Focused Male",
    "Chic, Pastels, Aesthetics-Focused Female"
]
# Filter valid timestamps
query_timestamps = [ts for ts in query_timestamps if ts in df['Timestamp'].values]

# Function to get top 10 similar respondents
def get_top_similar(query_idx, df, similarity_matrix, n=10):
    sim_scores = similarity_matrix[query_idx]
    top_indices = np.argsort(sim_scores)[::-1][1:n+1]  # Exclude self
    top_scores = sim_scores[top_indices]
    return [(df.iloc[idx]['Timestamp'], score) for idx, score in zip(top_indices, top_scores)]

# Store results
results = {}
for timestamp in query_timestamps:
    query_idx = df[df['Timestamp'] == timestamp].index[0]
    results[timestamp] = get_top_similar(query_idx, df, similarity_matrix)

# Print Results
x = 0
for timestamp, similar in results.items():
    print(f"\nTop 10 similar respondents for Query: {querys[x]} (Timestamp: {timestamp}):")
    x += 1
    for i, (ts, score) in enumerate(similar, 1):
        print(f"{i}. Timestamp {ts} (Score: {score:.2f})")

# Visualization
avg_scores = [np.mean([score for _, score in similar]) for similar in results.values()]
queries = ['Casual Female', 'Sporty Male', 'Chic Female'][:len(avg_scores)]

plt.figure(figsize=(8, 6))
plt.bar(queries, avg_scores, color=['#4CAF50', '#2196F3', '#FF9800'])
plt.title('Average Similarity Scores for Top 10 Matches')
plt.ylabel('Cosine Similarity Score')
plt.ylim(0, 1)
plt.savefig('similarity_scores.png')
plt.close()

# Save results to CSV
results_df = pd.DataFrame({
    'Query': [f"Query {i+1}: {q}" for i, q in enumerate(queries) for _ in range(10)],
    'Similar_Timestamp': [ts for similar in results.values() for ts, _ in similar],
    'Similarity_Score': [score for similar in results.values() for _, score in similar]
})
results_df.to_csv('similarity_results.csv', index=False)