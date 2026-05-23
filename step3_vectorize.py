import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load and clean inline (reusing step 2 logic)
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

df = pd.read_csv("C:\\Users\\THIS PC\\Downloads\\spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(clean_text)

# Split into train and test sets
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Preview results
print("=== Training set size ===")
print(X_train_tfidf.shape)

print("\n=== Test set size ===")
print(X_test_tfidf.shape)

print("\n=== Sample features (first 20 words TF-IDF learned) ===")
print(vectorizer.get_feature_names_out()[:20])