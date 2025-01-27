import nltk
from nltk.corpus import stopwords


def download_nlp_resources():
    """
    Downloads necessary NLTK resources like stopwords, punkt tokenizer, and wordnet if not already available.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

