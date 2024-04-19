import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy', '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused', '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',  'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once', 'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such', 't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom', 'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre", "youve", 'your', 'yours', 'yourself', 'yourselves']

def clean(text):
    processed_text = []
    word_lemm = WordNetLemmatizer() # Create Lemmatiser and Stemmer
    soup = BeautifulSoup(text, "html.parser") # Create soup

    # Defining regex patterns
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = "@[^\s]+"
    alpha_pattern = r"[^a-zA-Z0-9\s,']"
    seq_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"
    soup_pattern = r"\[[^]]*\]"

    text = text.lower()
    text = re.sub(url_pattern, ' URL', text) # Replace all URLs with 'URL'

    # Replace all emojis
    for emoji in emojis.keys():
        text = text.replace(emoji, "EMOJI" + emojis[emoji])

    text = re.sub(soup_pattern, '', soup.get_text()) # Soupy stuff
    text = re.sub(user_pattern, ' USER', text) # Replace @USERNAME to 'USER'
    text = re.sub(alpha_pattern, '', text) # Replace all non alphabets
    text = re.sub(seq_pattern, seq_replace_pattern, text) # Replace 3 or more consecutive letters by 2 letter.

    processed_text = ''
    for word in text.split():
        if len(word) > 1:
            word = word_lemm.lemmatize(word) # Lemmatising the word.
            processed_text += (word+' ')
        
    print(f"Processed text: {processed_text}")
    return processed_text