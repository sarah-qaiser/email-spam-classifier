import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# Load & Clean
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

# Split & Vectorize
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Metrics
print("=== Evaluation Results ===\n")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred, pos_label='spam'):.4f}")

print("\n=== Confusion Matrix ===\n")
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print(f"                Predicted Ham   Predicted Spam")
print(f"Actual Ham      {cm[0][0]}             {cm[0][1]}")
print(f"Actual Spam     {cm[1][0]}              {cm[1][1]}")