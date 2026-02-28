import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()

# Sample knowledge base
corpus = """
Hello, I am an AI chatbot.
I can answer questions about artificial intelligence.
Artificial intelligence is the simulation of human intelligence by machines.
Machine learning is a subset of artificial intelligence.
Natural language processing helps machines understand human language.
Python is a popular programming language for AI development.
NLTK and spaCy are popular NLP libraries in Python.
"""

# Preprocessing
sent_tokens = nltk.sent_tokenize(corpus)
word_tokens = nltk.word_tokenize(corpus)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "hey")
GREETING_RESPONSES = ["Hi!", "Hello!", "Hey there!", "Greetings!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generate chatbot response
def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    
    if score == 0:
        chatbot_response = "Sorry, I don't understand your question."
    else:
        chatbot_response = sent_tokens[idx]
        
    sent_tokens.pop()
    return chatbot_response

# Main chatbot loop
print("AI Chatbot: Hello! Ask me something about AI. (Type 'bye' to exit)")

while True:
    user_input = input("You: ")
    user_input = user_input.lower()
    
    if user_input == 'bye':
        print("AI Chatbot: Goodbye!")
        break
    
    elif greeting(user_input) is not None:
        print("AI Chatbot:", greeting(user_input))
    
    else:
        print("AI Chatbot:", response(user_input))