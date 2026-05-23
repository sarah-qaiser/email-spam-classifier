import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

# Train
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save both files
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("spam_model.pkl saved successfully!")
print("vectorizer.pkl saved successfully!")

# Verify by reloading and testing
loaded_model = joblib.load('spam_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

test_message = "WINNER! Claim your free prize now by calling 08000 now!"
cleaned = clean_text(test_message)
vectorized = loaded_vectorizer.transform([cleaned])
prediction = loaded_model.predict(vectorized)[0]
confidence = loaded_model.predict_proba(vectorized).max()

print(f"\nReload test passed!")
print(f"Message    : {test_message}")
print(f"Prediction : {prediction.upper()} (confidence: {confidence:.1%})")