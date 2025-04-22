import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Read Excel file
try:
    df = pd.read_csv('/Users/apple/Documents/VSC/python/Module2.csv')
except FileNotFoundError:
    print("Error: csv not found.")
    exit(1)
# Ensure required columns exist
if 'Sentiment' not in df.columns or 'Comment Author' not in df.columns:
    print("Error: Excel file must contain 'Sentiment' and 'Comment Author' columns.")
    exit(1)

user_sentiments = df.groupby('Comment Author')['Sentiment'].mean().reset_index()

# Create histogram with black background
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor('black')
ax.set_facecolor('black')

# Plot histogram
ax.hist(user_sentiments['Sentiment'], bins=20, color='skyblue', edgecolor='white', alpha=0.8)

# Customize axes
ax.set_title('Distribution of User Sentiment in All Communities', color='white', fontsize=14)
ax.set_xlabel('Average Sentiment Polarity (-1 to +1)', color='white', fontsize=12)
ax.set_ylabel('Number of Users', color='white', fontsize=12)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add grid for readability
ax.grid(True, color='gray', linestyle='--', alpha=0.3)

# Save and close plot
plt.savefig('sentiment_distribution.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()
print("Plot saved as sentiment_distribution.png")