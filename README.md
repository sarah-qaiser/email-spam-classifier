# Email Spam Classifier + Live Email Bot

An end-to-end NLP pipeline that classifies emails as spam or ham, 
deployed as a live email bot that automatically replies with the prediction.

## What It Does
The bot connects to a real email inbox via IMAP, reads incoming emails, 
runs them through the trained classification model, and sends back an 
automatic reply labeling the message as **Spam** or **Ham**.

## Project Structure
| File | Description |
|---|---|
| `step1_load_data.py` | Loads the SMS Spam Collection dataset |
| `step2_clean.py` | Text cleaning: lowercase, punctuation removal, stopwords, stemming |
| `step3_vectorize.py` | TF-IDF vectorization |
| `step4_train.py` | Model training |
| `step5_evaluate.py` | Evaluation metrics and results |
| `step6_save.py` | Saves trained model and vectorizer as .pkl files |
| `step7_email_bot.py` | Live email bot via IMAP/SMTP |
| `spam_model.pkl` | Saved trained model |
| `vectorizer.pkl` | Saved TF-IDF vectorizer |

## Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn, NLTK
- **TF-IDF** — text vectorization
- **IMAP/SMTP** — email reading and auto-reply
- **Pickle** — model serialization

## ML Pipeline
- Dataset: SMS Spam Collection (5,574 messages)
- Class distribution: ~87% ham / ~13% spam
- Preprocessing: lowercasing, punctuation removal, stopword filtering, stemming
- Vectorization: TF-IDF
- Prioritized **recall on spam** to minimize false negatives

## Screenshots
### Inbox
![Inbox](screenshot_inbox.png)

### Terminal Output
![Terminal](screenshot_terminal.png)

## How to Run
1. Install dependencies:
pip install scikit-learn pandas nltk
2. Run steps in order:
python step1_load_data.py
python step2_clean.py
python step3_vectorize.py
python step4_train.py
python step5_evaluate.py
python step6_save.py
3. Configure your email credentials in `step7_email_bot.py`
4. Run the bot:
python step7_email_bot.py
