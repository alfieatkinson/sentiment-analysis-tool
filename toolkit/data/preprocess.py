import os
import time

import re, time
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

import pandas as pd

import toolkit

class TextProcessor(object):
    """
    Class for processing text data.

    Attributes:
        lemmatiser (WordNetLemmatizer): The lemmatiser for word lemmatisation.
    """
    def __init__(self):
        """
        Initialise the TextProcessor object.
        """
        self.lemmatiser = WordNetLemmatizer() # Create lemmatiser and stemmer

    def clean(self, text: str) -> str:
        """
        Cleans the given text based on configured settings.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        toolkit.console(f"Unprocessed text - {text[:128] + '...' if len(text) > 128 else text}")

        # If configured, make text lowercase
        if toolkit.get_config('lowercase'):
            text = text.lower()

        text = self.preprocess(text) # Always apply basic substitution processing

        # If configured, make the text soupy
        if toolkit.get_config('soup'):
            text = self.soup(text)

        # If configured, lemmatise and stem the text
        if toolkit.get_config('lemmatise'):
            text = self.lemmatise(text)

        toolkit.console(f"Processed text - {text[:128] + '...' if len(text) > 128 else text}")

        return text

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the text by removing user and subreddit mentions, URLs, and special characters.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        x_mention_pattern = r'@\S{4,}'
        ampersand_pattern = r'&amp;' # Define the pattern for correcting '&amp;' to '&'
        url_pattern = r'((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)' # Define the pattern for URLs
        reddit_user_mention_pattern = r'/?u/\S+' # Define the pattern for reddit user mentions
        reddit_sub_mention_pattern = r'/?r/\S+' # Define the pattern for reddit user mentions
        reddit_url_pattern = r'\[(.*?)\]\(.*?\)' # Define the pattern for reddit URLs
        newline_pattern = r'(\r\n|\r|\n)'

        # Apply the patterns to the text
        text = re.sub(x_mention_pattern, 'USER', text)
        text = re.sub(ampersand_pattern, '&', text)
        text = re.sub(reddit_user_mention_pattern, 'USER', text)
        text = re.sub(reddit_sub_mention_pattern, 'SUBREDDIT', text)
        text = re.sub(reddit_url_pattern, '', text)
        text = re.sub(url_pattern, '', text)
        text = re.sub(newline_pattern, ' ', text)

        return text

    def lemmatise(self, text: str) -> str:
        """
        Lemmatises and stems the given text.

        Args:
            text (str): The text to be lemmatised and stemmed.

        Returns:
            str: The lemmatised and stemmed text.
        """
        lemmatised = '' # Initialise empty string to add lemmatised words to
        
        for word in text.split(): # Split the text into words
            if len(word) > 1:
                word = self.lemmatiser.lemmatize(word) # Lemmatising the word
                processed_text += (word + ' ') # Add the word to the processed text string

        return lemmatised
    
    def soup(self, text: str) -> str:
        """
        Applies BeautifulSoup to the given text.

        Args:
            text (str): The text to become soup.

        Returns:
            str: The soupy text.
        """
        soup = BeautifulSoup(text, 'html.parser') # Create soup
        soup_pattern = r'\[[^]]*\]'

        text = re.sub(soup_pattern, '', soup.get_text()) # Soupy stuff

        return text

def preprocess_dataset(path: str, dataset_size: int = 1600000) -> pd.DataFrame:
    """
    Preprocesses the CSV file by shuffling rows, replacing label values, and loading it into a pandas DataFrame.

    Args:
        path (str): Relative path to the CSV file.
        dataset_size (int): Size of the dataset to load.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the preprocessed dataset.
    """
    # Define the paths
    input_path = path + 'dataset.csv'
    output_path = path + 'processed-dataset.csv'

    # Read the preprocessed CSV file into a DataFrame
    toolkit.console(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path, names=['sentiment', 'timestamp', 'datestring', 'N/A', 'user', 'text'], encoding='latin-1')
    toolkit.console(f"Dataset loaded.")

    # Replace all label values of "4" with "1"
    df['sentiment'] = df['sentiment'].replace(4, 1)

    # Keep only the 'tweet' and 'score' columns
    df = df[['text', 'sentiment']]

    # Process the text
    toolkit.console("Processing text...")
    text_processor = TextProcessor()
    df['text'] = df['text'].apply(text_processor.clean)
    toolkit.console("Text processed.")

    # Shuffle the DataFrame
    toolkit.console("Shuffling dataset...")
    df = df.sample(frac=1).reset_index(drop=True)
    toolkit.console("Dataset shuffled.")

    # Save the preprocessed DataFrame to CSV
    toolkit.console("Saving dataset...")
    df.to_csv(output_path, index=False)
    toolkit.console("Dataset saved.")

    print()
    print(df)
    print()

    # Load the preprocessed CSV dataset into a pandas DataFrame
    return pd.read_csv(output_path, nrows=dataset_size)