import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import sys

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Example news articles
news_articles = [
    "The company reported excellent quarterly results.",
    "The market is facing a downturn due to economic uncertainties."
]

# Analyze sentiment
results = sentiment_pipeline(news_articles)

# Convert to 0-1 scale
sentiment_scores = []
for result in results:
    score = result['score']
    if result['label'] == 'POSITIVE':
        sentiment_scores.append(score)  # Positive scores are naturally 0-1
    else:
        sentiment_scores.append(1 - score)  # Convert negative to the same scale

print(sentiment_scores)

print("Aborting execution...")
sys.exit()

# 1. --------------- analyse correlation with taret series --------
# 2. Calculate Returns
price_data['Returns'] = price_data['Close'].pct_change()    # percentage changes

# 3. Combine Data
combined_data = pd.DataFrame({
    'Sentiment': sentiment_scores,
    'Returns': price_data['Returns']
})

# 4. Visualize the Data
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(combined_data['Sentiment'], label='Sentiment', color='blue')
plt.title('Sentiment Over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(combined_data['Returns'], label='Returns', color='green')
plt.title('Price Returns Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# 5. Correlation Analysis
correlation = combined_data['Sentiment'].corr(combined_data['Returns'])
print(f'Correlation between sentiment and returns: {correlation}')

# 6. Statistical Testing
correlation_coefficient, p_value = pearsonr(combined_data['Sentiment'].dropna(), combined_data['Returns'].dropna())
print(f'Pearson correlation coefficient: {correlation_coefficient}, p-value: {p_value}')

# 7. Rolling Correlation
rolling_correlation = combined_data['Sentiment'].rolling(window=30).corr(combined_data['Returns'])
plt.figure(figsize=(12, 6))
plt.plot(rolling_correlation, label='Rolling Correlation (30 days)', color='orange')
plt.title('Rolling Correlation Between Sentiment and Returns')
plt.legend()
plt.show()
