# Text Cleaning
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load data from Step 1
df = pd.read_csv("C:\\Users\\THIS PC\\Downloads\\spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Initialize tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Apply to every row
df['cleaned_text'] = df['text'].apply(clean_text)

# Preview the result
print("=== Before Cleaning ===")
print(df['text'][2])

print("\n=== After Cleaning ===")
print(df['cleaned_text'][2])