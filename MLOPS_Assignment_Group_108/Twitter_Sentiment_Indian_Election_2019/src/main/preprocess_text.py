import re
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords


def preprocess_text(input_text):
    """
    Preprocesses the given text by performing the following steps:
    - Converts text to lowercase
    - Removes URLs, user mentions, hashtags, special characters, and numbers
    - Tokenizes the text
    - Removes stopwords and applies lemmatization
    - Returns the cleaned and processed text as a single string
    """
    # Check if the input is a string
    if not isinstance(input_text, str):
        return ""  # Return an empty string for non-string entries or NaN

    # Convert to lowercase
    input_text = input_text.lower()

    # Remove URLs
    input_text = re.sub(r'http\S+|www\S+|https\S+', '', input_text, flags=re.MULTILINE)

    # Remove user mentions and hashtags
    input_text = re.sub(r'@\w+|#\w+', '', input_text)

    # Remove special characters, numbers, and punctuations
    input_text = re.sub(r'[^a-zA-Z\s]', '', input_text)

    # Tokenize text
    tokens = word_tokenize(input_text)

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and apply lemmatization
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join the words back into one string
    return ' '.join(processed_tokens)
