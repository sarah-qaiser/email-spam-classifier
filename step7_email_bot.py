import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time

#Email Credentials
EMAIL_ADDRESS = "sarahqaiser259@gmail.com"
APP_PASSWORD  = "osabsmplujablklw"

# Load saved model and vectorizer
model      = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Same cleaning function from Step 2 
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

# Classify a message 
def classify(text):
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction  = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized).max()
    return prediction, confidence

# Send a reply 
def send_reply(to_address, original_subject, prediction, confidence):
    msg = MIMEMultipart()
    msg['From']    = EMAIL_ADDRESS
    msg['To']      = to_address
    msg['Subject'] = f"Re: {original_subject}"

    body = f"""Hello,

Your message has been automatically analysed by our Text Classification Bot.

Result     : {prediction.upper()}
Confidence : {confidence:.1%}

This is an automated reply. No human has read your message.

-- Text Classification Bot (ML Assignment 4)
"""
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_ADDRESS, APP_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_address, msg.as_string())
    print(f"    ↳ Reply sent to {to_address}")

# Main bot loop
def run_bot():
    print("=" * 50)
    print("   Email Classification Bot — RUNNING")
    print("=" * 50)

    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(EMAIL_ADDRESS, APP_PASSWORD)
    mail.select('inbox')

    # Search for unread emails
    TEST_SENDER = "sarahqaisersahaf@gmail.com"  
    _, message_numbers = mail.search(None, f'(UNSEEN FROM "{TEST_SENDER}")')
    email_ids = message_numbers[0].split()

    if not email_ids:
        print("📭 No unread emails found.")
    else:
        print(f"📬 Found {len(email_ids)} unread email(s).\n")

        for num in email_ids:
            _, msg_data = mail.fetch(num, '(RFC822)')
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Extract sender and subject
            sender  = msg['From']
            subject = msg['Subject'] or "(no subject)"

            # Extract body text
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

            print(f"📧 From    : {sender}")
            print(f"   Subject : {subject}")

            # Classify
            text_to_classify = subject + " " + body
            prediction, confidence = classify(text_to_classify)

            print(f"   Result  : {prediction.upper()} (confidence: {confidence:.1%})")

            # Send reply
            send_reply(sender, subject, prediction, confidence)
            print()

    mail.logout()
    print("\n✅ Bot finished. Run again to check new emails.")

# Run
run_bot()