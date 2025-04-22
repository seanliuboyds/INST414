import praw
import csv
from datetime import datetime
from textblob import TextBlob
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np

# Gets sentiment of comment from textblob
def get_sentiment(comment):
    try:
        if isinstance(comment, str) and comment.strip() != "[removed]":
            return TextBlob(comment).sentiment.polarity
        return 0.0
    except Exception as e:
        print(f"Error analyzing comment: {e}")
        return 0.0

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id="AOQArPj5rnUTVEl37SG6uA",
    client_secret="ibW0TSFB5DnZAM9eunOpd-AdOuT8Gg",
    user_agent="Conservative Discourse Tracker by u/Jealous_Fish_2480",
)

# Define subreddit / change this and compare results
subreddits = ["Conservative"]
keywords = ["QAnon", "deep state", "election fraud", "false flag", "cabal", "new world order", "stolen election", "pizzagate", "pizza gate", "globalist", "soros", "fauci", "biden crime family"]
max_entries = 600

# Open CSV file to store results
with open("conspiracy_replies.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Subreddit", "Post Title", "Post Score", "Comment", "Comment Author", "Comment Score", "Created", "Keyword", "Sentiment", "Parent Author"])
    entry_count = 0
    sub_name = subreddits[0]

    try:
        subreddit = reddit.subreddit(sub_name)
        for keyword in keywords:
            if entry_count >= max_entries:
                break
            for submission in subreddit.search(keyword, limit=5):
                if entry_count >= max_entries:
                    break
                try:
                    created = datetime.fromtimestamp(submission.created_utc)
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if entry_count >= max_entries:
                            break
                        if any(k.lower() in comment.body.lower() for k in keywords):
                            author = str(comment.author) if comment.author else "None"
                            parent = comment.parent()
                            parent_author = str(parent.author) if hasattr(parent, 'author') and parent.author else "None"
                            if author != "None" and parent_author != "None":
                                writer.writerow([
                                    sub_name,
                                    submission.title,
                                    submission.score,
                                    comment.body,
                                    author,
                                    comment.score,
                                    created,
                                    keyword,
                                    get_sentiment(comment.body),
                                    parent_author
                                ])
                                entry_count += 1
                                # print(f"Entry {entry_count} (r/{sub_name}): {comment.body[:50]}...")
                except Exception as e:
                    print(f"Error processing post in r/{sub_name}: {e}")
    except Exception as e:
        print(f"Error in r/{sub_name}: {e}")

print(f"Collected {entry_count} entries. Data saved to conspiracy_replies.csv")

# Load CSV
try:
    df = pd.read_csv("conspiracy_replies.csv")
except FileNotFoundError:
    print("Error: conspiracy_replies.csv not found.")
    exit(1)

# Create directed graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in df.iterrows():
    author = row['Comment Author']
    parent_author = row['Parent Author']
    sentiment = row['Sentiment']
    if author in G:
        G.nodes[author]['sentiments'].append(sentiment)
    else:
        G.add_node(author, sentiments=[sentiment])
    if parent_author not in G:
        G.add_node(parent_author, sentiments=[])
    G.add_edge(author, parent_author)  # Directed: author -> parent_author (reply to)

# Compute average sentiment per user
for node in G.nodes:
    sentiments = G.nodes[node]['sentiments']
    G.nodes[node]['avg_sentiment'] = sum(sentiments) / len(sentiments) if sentiments else 0.0

# Compute in-degree centrality
centrality = nx.in_degree_centrality(G)

# Identify influential users (high in-degree centrality)
influencers = [
    (node, centrality[node], G.nodes[node]['avg_sentiment'])
    for node in G.nodes
    if centrality[node] > 0.01
]
influencers.sort(key=lambda x: x[1], reverse=True)
print("Length of Influencers: " + str(len(influencers)))

# Print top influencers
print("Top Influential Users (High In-Degree Centrality):")
for node, cent, sent in influencers[:10]:
    print(f"User: {node}, In-Degree Centrality: {cent:.3f}, Avg Sentiment: {sent:.3f}")
    comment = df[df['Comment Author'] == node]['Comment'].iloc[0] if node in df['Comment Author'].values else "N/A"

# Visualize
fig, ax = plt.subplots(figsize=(12, 10))
colors = [G.nodes[node]['avg_sentiment'] for node in G.nodes]
sizes = [centrality[node] * 100 for node in G.nodes]

norm = Normalize(vmin=min(colors, default=-1), vmax=max(colors, default=1))
mappable = ScalarMappable(norm=norm, cmap=plt.cm.RdYlBu)
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label('Avg Sentiment', color='black')
cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
cbar.outline.set_edgecolor('black')

pos = nx.spring_layout(G, k=0.3)

nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.RdYlBu, node_size=70, edgecolors='black', linewidths=1, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.8, width=0.5, arrows=True, ax=ax)

ax.set_title('User Reply Network in r/conspiracy: In-Degree Centrality and Sentiment', color='white')
plt.savefig('reply_network_conspiracy_black.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

print("Saved in current directory!")