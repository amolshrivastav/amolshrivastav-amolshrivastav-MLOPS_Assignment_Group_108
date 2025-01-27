import wordcloud as WordCloud
import matplotlib.pyplot as plt


def perform_twitter_eda(df):
    """
    Performs exploratory data analysis (EDA) on a DataFrame containing Twitter data.
    Includes sentiment distribution analysis with bar and pie charts.
    """
    # Drop rows with missing values in 'clean_text' or 'category'
    df = df.dropna(subset=['clean_text', 'category'])

    # Ensure 'category' is treated as an integer (sentiment labels)
    df['category'] = df['category'].astype(int)

    # Analyze sentiment distribution
    sentiment_distribution = df['category'].value_counts().sort_index()

    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot sentiment distribution as a bar chart
    sentiment_distribution.plot(kind='bar', color=['red', 'gray', 'green'], ax=ax1)
    ax1.set_title('Sentiment Distribution')
    ax1.set_xlabel('Sentiment Category')
    ax1.set_ylabel('Number of Tweets')
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Negative', 'Neutral', 'Positive'], rotation=0)

    # Annotate each bar with count and percentage
    for i, count in enumerate(sentiment_distribution):
        percentage = count / sentiment_distribution.sum() * 100
        ax1.text(i, count + 0.05, f'{count} ({percentage:.1f}%)', ha='center', va='bottom')

    # Plot sentiment distribution as a pie chart
    ax2.pie(
        sentiment_distribution,
        labels=['Negative', 'Neutral', 'Positive'],
        autopct='%1.1f%%',
        colors=['red', 'gray', 'green'],
        startangle=90
    )
    ax2.set_title('Sentiment Distribution (Pie Chart)')

    # Display the plots
    plt.tight_layout()
    plt.show()


def generate_wordcloud(text, title):
    """
    Generates and displays a word cloud for a given text and title.
    """
    wc = WordCloud.WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()