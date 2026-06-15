
# FraudShield - Email and SMS Fraud Detector

import os
import string
import pickle
import sqlite3
import random
import re
import threading
import time
import html
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# PART 1 - SETTINGS

MODEL_FILE    = "fraud_model.pkl"
DATABASE_FILE = "fraud_history.db"
DATASET_FILE  = "spam.csv"
SPAM_THRESHOLD = 0.40   # topic detector threshold for specific scam patterns
MODEL_SPAM_THRESHOLD = 0.85   # tree model threshold for general spam probability
MODEL_VERSION = 5

SCAM_TOPICS = {
    "Phishing Email": {
        "weight": 0.30,
        "base_score": 0.68,
        "keywords": [
            "verify", "login", "password", "credential", "account locked",
            "account suspended", "restore access", "security alert", "confirm identity",
            "click here", "update your account"
        ],
    },
    "Banking Scam": {
        "weight": 0.35,
        "base_score": 0.70,
        "keywords": [
            "bank", "sbi", "hdfc", "icici", "axis", "paytm", "paypal", "kyc",
            "aadhaar", "pan", "ifsc", "cvv", "card number", "account number",
            "refund", "tax refund", "wire transfer", "blocked"
        ],
    },
    "OTP Scam": {
        "weight": 0.40,
        "base_score": 0.72,
        "keywords": [
            "otp", "one time password", "verification code", "share code",
            "card pin", "upi pin", "confirm otp", "send otp"
        ],
    },
    "Malicious URL": {
        "weight": 0.35,
        "base_score": 0.78,
        "keywords": [
            "http", "https", "www", ".xyz", ".top", ".click", ".link", "bit.ly",
            "tinyurl", "shorturl", "login at", "open link", "click the link"
        ],
    },
    "Fake Job Scam": {
        "weight": 0.35,
        "base_score": 0.74,
        "keywords": [
            "job", "work from home", "earn daily", "salary", "registration fee",
            "processing fee", "interview fee", "joining fee", "part time",
            "no experience", "guaranteed income", "whatsapp hr", "telegram"
        ],
    },
    "Loan Offer Spam": {
        "weight": 0.42,
        "base_score": 0.76,
        "keywords": [
            "loan", "personal loan", "instant loan", "cash loan", "pre approved",
            "pre-approved", "no documents", "low emi", "zero processing fee",
            "credit limit", "credit card offer", "quick approval", "apply now",
            "disbursal", "lowest interest", "cibil"
        ],
    },
    "Promotional Ad": {
        "weight": 0.40,
        "base_score": 0.70,
        "keywords": [
            "sale", "discount", "offer", "limited time", "buy now", "free gift",
            "cashback", "coupon", "promo code", "flat off", "deal", "exclusive",
            "subscribe", "unsubscribe", "recharge", "shopping", "valid till"
        ],
    },
}

TRAINING_EXAMPLES = [
    ("spam", "Your email account is suspended verify password at secure-login-update.xyz now"),
    ("spam", "Security alert confirm your identity and restore access using this link"),
    ("spam", "Mailbox storage full login now to confirm identity or incoming emails will stop"),
    ("spam", "Microsoft account unusual sign in detected verify credentials at account-helpdesk-login.com"),
    ("spam", "Your email password expires today update your account to avoid permanent suspension"),
    ("spam", "Bank KYC pending update Aadhaar PAN card number CVV and IFSC immediately"),
    ("spam", "SBI account blocked complete KYC today or your net banking will stop"),
    ("spam", "HDFC debit card blocked verify card number CVV and expiry date urgently"),
    ("spam", "Your bank refund is pending share account number IFSC and Aadhaar to receive money"),
    ("spam", "UPI transaction failed call support and provide PIN to reverse bank transfer"),
    ("spam", "Share OTP to cancel unauthorized bank transfer from your account"),
    ("spam", "Your one time password is required by bank support send OTP now"),
    ("spam", "Forward the verification code to reactivate your wallet within 10 minutes"),
    ("spam", "Send OTP and UPI PIN to stop suspicious debit from your savings account"),
    ("spam", "Customer care needs the SMS code to approve refund immediately"),
    ("spam", "Click http://secure-bank-login.xyz to verify your account before midnight"),
    ("spam", "Open bit.ly/free-reward and enter your password to claim prize"),
    ("spam", "Visit http://192.168.1.5/login and verify wallet secret phrase today"),
    ("spam", "Track package at tinyurl.com/pay-fee-release and pay small delivery fee"),
    ("spam", "Login at paypal-secure-update.top to avoid account closure"),
    ("spam", "Work from home job earn 5000 daily pay registration fee on WhatsApp"),
    ("spam", "Fake HR selected you for job send processing fee and documents today"),
    ("spam", "Part time job no experience salary 3000 per day join telegram and pay interview fee"),
    ("spam", "HR recruitment offer guaranteed income send joining fee to receive appointment letter"),
    ("spam", "Online data entry vacancy pay training fee and start earning today"),
    ("spam", "Pre approved personal loan available with low EMI apply now no documents required"),
    ("spam", "Instant cash loan approved quick disbursal click link to claim offer today"),
    ("spam", "Credit card limit increased apply now with zero processing fee and lowest interest"),
    ("spam", "Personal loan approved without CIBIL check submit documents on WhatsApp"),
    ("spam", "Need urgent cash loan get 2 lakh in 5 minutes no salary slip required"),
    ("spam", "Exclusive sale flat 70 percent discount buy now limited time offer"),
    ("spam", "Recharge cashback coupon valid till midnight click to claim free gift"),
    ("spam", "Mega shopping deal buy now use promo code and get free gift today"),
    ("spam", "Limited time offer subscribe now for discount coupon and cashback"),
    ("spam", "Flash sale valid till midnight click link to claim exclusive reward"),
    ("ham", "Your bank statement is available in the official app"),
    ("ham", "Your OTP is 123456 do not share it with anyone"),
    ("ham", "Your job interview is scheduled for Monday with the HR team"),
    ("ham", "Please use the company portal to update your password"),
    ("ham", "Your loan EMI receipt for this month is attached for your records"),
    ("ham", "The bank branch appointment is confirmed for 11am tomorrow"),
    ("ham", "Your credit card statement is ready in the official banking app"),
    ("ham", "The marketing team will review the promotional campaign draft today"),
    ("ham", "The sale forecast report is ready for the quarterly business review"),
    ("ham", "Please share the job description document before the hiring meeting"),
    ("ham", "Your package tracking link is available inside the official shopping app"),
]


# PART 2 - TEXT CLEANING

def clean_text(text):
    text = text.lower()
    words = text.split()
    clean_words = []
    for word in words:
        word = word.strip(string.punctuation)
        if len(word) > 2:
            clean_words.append(word)
    return " ".join(clean_words)


def find_urls(text):
    return re.findall(r"(?:https?://|www\.)[^\s]+|[a-zA-Z0-9.-]+\.(?:com|net|org|in|co|xyz|top|click|link|info|ru)[^\s]*", text)


def detect_malicious_urls(text):
    urls = find_urls(text)
    indicators = []
    suspicious_tlds = (".xyz", ".top", ".click", ".link", ".ru", ".info")
    shorteners = ("bit.ly", "tinyurl.com", "shorturl.at", "t.co", "goo.gl", "is.gd")
    trusted_brands = ("bank", "sbi", "hdfc", "icici", "axis", "paypal", "amazon", "google", "microsoft")

    for raw_url in urls:
        url = raw_url.strip(".,;:!?()[]{}<>")
        parsed = urlparse(url if "://" in url else "https://" + url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if any(host.endswith(tld) for tld in suspicious_tlds):
            indicators.append("suspicious URL domain: " + host)
        if any(shortener in host for shortener in shorteners):
            indicators.append("shortened URL: " + host)
        if re.search(r"\d+\.\d+\.\d+\.\d+", host):
            indicators.append("IP address URL: " + host)
        if "@" in url:
            indicators.append("URL contains @ symbol")
        if "-" in host and any(brand in host for brand in trusted_brands):
            indicators.append("brand-like hyphenated domain: " + host)
        if any(word in path for word in ["login", "verify", "kyc", "password", "wallet"]):
            indicators.append("sensitive action in URL path")

    return indicators


def detect_scam_topics(full_text):
    text = full_text.lower()
    topic_scores = {}
    topic_indicators = {}

    for topic, config in SCAM_TOPICS.items():
        matches = []
        for keyword in config["keywords"]:
            if keyword in text:
                matches.append(keyword)
        if topic == "OTP Scam" and matches:
            otp_abuse_words = [
                "send", "confirm", "provide", "forward", "cancel",
                "unauthorized", "support", "customer care", "upi pin", "card pin"
            ]
            asks_to_share = "share" in text and "do not share" not in text
            safe_otp_notice = "do not share" in text and not asks_to_share and not any(word in text for word in otp_abuse_words)
            if safe_otp_notice:
                matches = []
        if matches:
            topic_scores[topic] = min(0.94, config.get("base_score", config["weight"]) + (len(matches) * 0.04))
            topic_indicators[topic] = matches[:6]

    url_indicators = detect_malicious_urls(full_text)
    if url_indicators:
        topic_scores["Malicious URL"] = max(
            topic_scores.get("Malicious URL", 0),
            min(0.96, SCAM_TOPICS["Malicious URL"]["base_score"] + (len(url_indicators) * 0.05))
        )
        topic_indicators["Malicious URL"] = url_indicators[:6]

    urgency_words = ["urgent", "immediately", "within 24 hours", "final warning", "blocked", "suspended", "expires"]
    sensitive_words = ["password", "otp", "cvv", "pin", "aadhaar", "pan", "bank details", "account number"]
    has_urgency = any(word in text for word in urgency_words)
    has_sensitive = any(word in text for word in sensitive_words)

    if has_urgency and has_sensitive:
        for topic in ["Phishing Email", "Banking Scam", "OTP Scam"]:
            if topic in topic_scores:
                topic_scores[topic] = min(0.97, topic_scores[topic] + 0.12)
                topic_indicators[topic].append("urgent request for sensitive information")

    if not topic_scores:
        return "General Spam", [], 0.0

    scam_type = max(topic_scores, key=topic_scores.get)
    indicators = topic_indicators.get(scam_type, [])
    return scam_type, indicators, topic_scores[scam_type]


def count_keywords(text, keywords):
    return sum(1 for keyword in keywords if keyword in text)


def calibrate_display_score(model_probability, topic_score, is_spam):
    if is_spam:
        score = max(model_probability, topic_score)
        if model_probability >= MODEL_SPAM_THRESHOLD and topic_score >= SPAM_THRESHOLD:
            score = min(0.99, score + 0.06)
        return min(0.99, max(score, 0.70))

    return min(model_probability, 0.25)


def extract_text_features(texts):
    rows = []
    urgency_words = ["urgent", "immediately", "within 24 hours", "final warning", "blocked", "suspended", "expires", "today"]
    sensitive_words = ["password", "otp", "cvv", "pin", "aadhaar", "pan", "bank details", "account number", "ifsc"]
    money_words = ["lakh", "rupees", "rs", "cash", "emi", "refund", "salary", "discount", "cashback", "free"]

    for value in texts:
        raw = "" if value is None else str(value)
        lower = raw.lower()
        words = re.findall(r"[a-zA-Z0-9]+", raw)
        urls = find_urls(raw)
        malicious_url_count = len(detect_malicious_urls(raw))
        digit_count = sum(ch.isdigit() for ch in raw)
        upper_count = sum(ch.isupper() for ch in raw)
        char_count = max(len(raw), 1)

        topic_counts = []
        for config in SCAM_TOPICS.values():
            topic_counts.append(count_keywords(lower, config["keywords"]))

        rows.append([
            len(raw),
            len(words),
            len(urls),
            malicious_url_count,
            digit_count / char_count,
            upper_count / char_count,
            raw.count("!"),
            raw.count("$") + raw.count("₹"),
            len(re.findall(r"\b\d{4,}\b", raw)),
            count_keywords(lower, urgency_words),
            count_keywords(lower, sensitive_words),
            count_keywords(lower, money_words),
            int(bool(re.search(r"\b(?:call|whatsapp|telegram)\b", lower))),
            int(bool(re.search(r"\b(?:apply now|click|claim|buy now|login|verify)\b", lower))),
            *topic_counts,
        ])

    return np.asarray(rows, dtype=float)


def build_classifier():
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.2,
            eval_metric="logloss",
            random_state=42,
        ), "XGBoost"

    return RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ), "RandomForest"


# ============================================================
# PART 3 - TRAIN AND SAVE MODEL
# Uses the spam.csv dataset plus extra scam examples
# Algorithm: TF-IDF + XGBoost, with RandomForest fallback
# ============================================================

def train_model():
    st.info("Training AI model from spam.csv dataset. Please wait...")
   
    # load the dataset
    df = pd.read_csv(DATASET_FILE, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    extra_df = pd.DataFrame(TRAINING_EXAMPLES, columns=["label", "message"])
    df = pd.concat([df, extra_df], ignore_index=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna()

    # split into train and test sets
    X = df["message"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    classifier, algorithm = build_classifier()

    # build the ML pipeline
    # Word TF-IDF, character n-grams, and engineered fraud signals are combined.
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                max_features=8000,
                ngram_range=(1, 3),
                sublinear_tf=True,
            )),
            ("char_tfidf", TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("engineered", FunctionTransformer(extract_text_features, validate=False)),
        ])),
        ("model", classifier),
    ])

    # train the model
    pipeline.fit(X_train, y_train)

    # check accuracy
    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=["Legitimate", "Spam"])
    cm       = confusion_matrix(y_test, y_pred)

    # save accuracy info to show in UI
    info = {
        "accuracy"   : round(accuracy * 100, 2),
        "report"     : report,
        "cm"         : cm.tolist(),
        "trained_at" : datetime.now().strftime("%d %b %Y %H:%M"),
        "algorithm"  : algorithm,
        "feature_engineering": "word TF-IDF + character n-grams + URL/OTP/banking/loan/promo/job signals",
        "total_data" : len(df),
        "extra_training_examples": len(TRAINING_EXAMPLES),
        "spam_count" : int(df["label_num"].sum()),
        "ham_count"  : int(len(df) - df["label_num"].sum()),
    }

    # save the model to disk
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"version": MODEL_VERSION, "pipeline": pipeline, "info": info}, f)

    return pipeline, info


def load_model():
    with open(MODEL_FILE, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("version") != MODEL_VERSION:
        os.remove(MODEL_FILE)
        return None, None
    model = obj.get("pipeline")
    info = obj.get("info")
    # safety check - make sure it is a real model not a broken file
    if not hasattr(model, "predict_proba"):
        os.remove(MODEL_FILE)
        return None, None
    return model, info


def get_model():
    # if model file exists load it, otherwise train from scratch
    if os.path.exists(MODEL_FILE):
        model, info = load_model()
        if model is not None:
            return model, info
    # no model found or broken - train new one
    if not os.path.exists(DATASET_FILE):
        st.error("spam.csv not found. Please put spam.csv in the same folder as app.py")
        st.stop()
    return train_model()


# ============================================================
# PART 4 - PREDICT IF A MESSAGE IS FRAUD OR LEGITIMATE
# ============================================================

def predict_message(model, text, sender="", subject=""):
    # combine all parts of the message for better accuracy
    full_text  = subject + " " + sender + " " + text
    model_probability = float(model.predict_proba([full_text])[0][1])
    scam_type, indicators, topic_boost = detect_scam_topics(full_text)
    is_spam    = model_probability >= MODEL_SPAM_THRESHOLD or topic_boost >= SPAM_THRESHOLD
    display_probability = calibrate_display_score(model_probability, topic_boost, is_spam)

    # decide risk level based on probability
    if not is_spam:
        risk = "SAFE"
    elif display_probability >= 0.80:
        risk = "HIGH"
    elif display_probability >= 0.55:
        risk = "MEDIUM"
    elif display_probability >= SPAM_THRESHOLD:
        risk = "LOW"
    else:
        risk = "SAFE"

    return {
        "is_spam"     : is_spam,
        "probability" : round(display_probability * 100, 1),
        "model_probability": round(model_probability * 100, 1),
        "scam_type"   : scam_type if is_spam else "None",
        "indicators"  : ", ".join(indicators) if indicators else "No strong scam topic indicators",
        "risk"        : risk,
        "label"       : "FRAUD" if is_spam else "LEGITIMATE",
    }


# ============================================================
# PART 5 - DATABASE FUNCTIONS
# SQLite stores all scanned messages so user can review history
# ============================================================

def setup_database():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sender      TEXT,
            subject     TEXT,
            body        TEXT,
            msg_type    TEXT,
            label       TEXT,
            probability REAL,
            risk        TEXT,
            scam_type   TEXT DEFAULT 'None',
            indicators  TEXT DEFAULT '',
            scanned_at  TEXT,
            delete_at   TEXT,
            is_deleted  INTEGER DEFAULT 0
        )
    """)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if "scam_type" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN scam_type TEXT DEFAULT 'None'")
    if "indicators" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN indicators TEXT DEFAULT ''")
    conn.commit()
    conn.close()


def save_to_database(sender, subject, body, msg_type, label, probability, risk, scam_type="None", indicators=""):
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # fraud messages are deleted after 24 hours
    delete_at = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S") if label == "FRAUD" else None

    conn = sqlite3.connect(DATABASE_FILE)
    conn.execute("""
        INSERT INTO messages (sender, subject, body, msg_type, label, probability, risk, scam_type, indicators, scanned_at, delete_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (sender, subject, body, msg_type, label, probability, risk, scam_type, indicators, now, delete_at))
    conn.commit()
    conn.close()


def load_history(filter_type="All", search_text=""):
    conn  = sqlite3.connect(DATABASE_FILE)
    query = """
        SELECT id, sender, subject, body, msg_type, label, probability, risk,
               scanned_at, delete_at, scam_type, indicators
        FROM messages WHERE is_deleted=0
    """

    if filter_type == "Fraud Only":
        query += " AND label='FRAUD'"
    elif filter_type == "Legitimate Only":
        query += " AND label='LEGITIMATE'"

    if search_text:
        query += " AND (sender LIKE ? OR subject LIKE ? OR body LIKE ? OR scam_type LIKE ? OR indicators LIKE ?)"
        pattern = "%" + search_text + "%"
        rows = conn.execute(query + " ORDER BY scanned_at DESC", (pattern, pattern, pattern, pattern, pattern)).fetchall()
    else:
        rows = conn.execute(query + " ORDER BY scanned_at DESC").fetchall()

    conn.close()
    return rows


def get_counts():
    conn    = sqlite3.connect(DATABASE_FILE)
    total   = conn.execute("SELECT COUNT(*) FROM messages WHERE is_deleted=0").fetchone()[0]
    fraud   = conn.execute("SELECT COUNT(*) FROM messages WHERE label='FRAUD' AND is_deleted=0").fetchone()[0]
    legit   = conn.execute("SELECT COUNT(*) FROM messages WHERE label='LEGITIMATE' AND is_deleted=0").fetchone()[0]
    deleted = conn.execute("SELECT COUNT(*) FROM messages WHERE is_deleted=1").fetchone()[0]
    conn.close()
    return total, fraud, legit, deleted


def delete_expired_fraud():
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DATABASE_FILE)
    cur  = conn.execute("""
        UPDATE messages SET is_deleted=1
        WHERE label='FRAUD' AND delete_at IS NOT NULL AND delete_at <= ? AND is_deleted=0
    """, (now,))
    count = cur.rowcount
    conn.commit()
    conn.close()
    return count


def get_daily_counts():
    conn   = sqlite3.connect(DATABASE_FILE)
    dates  = []
    frauds = []
    legits = []
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        f   = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE label='FRAUD' AND scanned_at LIKE ?",
            (day + "%",)
        ).fetchone()[0]
        l   = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE label='LEGITIMATE' AND scanned_at LIKE ?",
            (day + "%",)
        ).fetchone()[0]
        dates.append(day[5:])   # show only MM-DD
        frauds.append(f)
        legits.append(l)
    conn.close()
    return dates, frauds, legits


def get_scam_type_counts():
    conn = sqlite3.connect(DATABASE_FILE)
    rows = conn.execute("""
        SELECT scam_type, COUNT(*) FROM messages
        WHERE label='FRAUD' AND is_deleted=0
        GROUP BY scam_type
        ORDER BY COUNT(*) DESC
    """).fetchall()
    conn.close()
    return rows


# ============================================================
# PART 6 - DEMO MESSAGES FOR SIMULATION
# These simulate real incoming emails and SMS
# In real deployment these would come from Gmail API or Twilio
# ============================================================

DEMO_MESSAGES = [
    {
        "sender"  : "alerts@bank-secure-verify.net",
        "subject" : "URGENT Account suspended",
        "body"    : "Dear customer your bank account has been locked. Verify credentials at secure-bank-login.xyz immediately to restore access.",
        "type"    : "Email",
    },
    {
        "sender"  : "1-888-555-0192",
        "subject" : "SMS",
        "body"    : "Congratulations you have won a 500 dollar Amazon gift card. Act now limited time offer. Click the link to claim before it expires tonight.",
        "type"    : "SMS",
    },
    {
        "sender"  : "rahul.sharma@gmail.com",
        "subject" : "Team lunch tomorrow",
        "body"    : "Hey are we still on for lunch tomorrow at 1pm? Let me know if it works for you.",
        "type"    : "Email",
    },
    {
        "sender"  : "irs-refund@gov-tax-refund.com",
        "subject" : "IRS Tax Refund Action Required",
        "body"    : "You are eligible for a 2840 dollar refund. Provide your SSN and bank details to process the wire transfer within 24 hours.",
        "type"    : "Email",
    },
    {
        "sender"  : "dev@github.com",
        "subject" : "Pull request review requested",
        "body"    : "Rohit requested your review on pull request 312. Add JWT authentication middleware. The changes look clean overall.",
        "type"    : "Email",
    },
    {
        "sender"  : "support@paypa1-secure.com",
        "subject" : "PayPal account verification needed",
        "body"    : "Valued member your PayPal account expires soon. Enter your password and CVV at the link below to avoid permanent suspension.",
        "type"    : "Email",
    },
    {
        "sender"  : "91-800-555-0147",
        "subject" : "SMS",
        "body"    : "You have been selected for a FREE iPhone 16. Limited winners only. Click the link before offer expires tonight.",
        "type"    : "SMS",
    },
    {
        "sender"  : "LOANEX",
        "subject" : "SMS",
        "body"    : "Pre approved personal loan up to 5 lakh with low EMI and no documents required. Apply now for quick disbursal.",
        "type"    : "SMS",
    },
    {
        "sender"  : "SHOPAD",
        "subject" : "Mega Sale",
        "body"    : "Exclusive sale today. Flat 70 percent discount, cashback coupon and free gift. Buy now before the offer expires.",
        "type"    : "Email",
    },
    {
        "sender"  : "crypto@bitcoin-double.com",
        "subject" : "Double your Bitcoin guaranteed",
        "body"    : "URGENT investment opportunity. Send 0.1 BTC and receive 0.2 BTC back guaranteed. Offer expires tonight. Act immediately.",
        "type"    : "Email",
    },
    {
        "sender"  : "noreply@irctc.co.in",
        "subject" : "Booking confirmed PNR 1234567890",
        "body"    : "Your IRCTC booking is confirmed for 2 passengers on Rajdhani Express. Train departs at 6am. PNR details are in this email.",
        "type"    : "Email",
    },
    {
        "sender"  : "kyc-sbi@sbi-kyc-update.net",
        "subject" : "SBI KYC Update Required",
        "body"    : "Your SBI account has been blocked due to incomplete KYC. Update your Aadhaar and PAN card details within 24 hours to avoid permanent account suspension.",
        "type"    : "Email",
    },
    {
        "sender"  : "priya.college@gmail.com",
        "subject" : "SMS",
        "body"    : "Hi this is Priya from college. Are you coming to the reunion this Saturday? Please confirm by Friday.",
        "type"    : "SMS",
    },
    {
        "sender"  : "91-99900-11111",
        "subject" : "SMS",
        "body"    : "Your mobile number won 3 lakh rupees in BSNL lucky draw. Call immediately to claim your prize and share your bank details.",
        "type"    : "SMS",
    },
    {
        "sender"  : "91-88000-22334",
        "subject" : "SMS",
        "body"    : "Bank support here. Share the OTP you received to cancel an unauthorized transfer from your account immediately.",
        "type"    : "SMS",
    },
    {
        "sender"  : "alerts@secure-bank-login.xyz",
        "subject" : "Wallet suspended",
        "body"    : "Final warning. Login at http://secure-bank-login.xyz/verify-password now to prevent wallet suspension.",
        "type"    : "Email",
    },
    {
        "sender"  : "hr@fastcareer-offer.com",
        "subject" : "Immediate job selection",
        "body"    : "You are selected for a work from home job. Earn 5000 daily. Pay registration fee on WhatsApp to start today.",
        "type"    : "Email",
    },
    {
        "sender"  : "hr@infosys.com",
        "subject" : "Internship joining confirmation",
        "body"    : "Dear candidate we are pleased to confirm your internship start date as June 1st. Please bring original documents on the first day.",
        "type"    : "Email",
    },
    {
        "sender"  : "income-tax@gov-refund-india.net",
        "subject" : "Income tax refund pending",
        "body"    : "Your income tax refund of 18500 is ready. Share your bank account IFSC and Aadhaar number to receive the amount immediately.",
        "type"    : "Email",
    },
    {
        "sender"  : "manager@mycompany.com",
        "subject" : "Project update meeting",
        "body"    : "Hi team the client approved our proposal. Please update your tasks in Jira. We will review progress on Friday at 3pm.",
        "type"    : "Email",
    },
]


# ============================================================
# PART 7 - BACKGROUND AUTO DELETE THREAD
# Runs every 60 seconds to delete fraud messages older than 24h
# ============================================================

stop_thread = threading.Event()


def auto_delete_loop():
    while not stop_thread.is_set():
        delete_expired_fraud()
        stop_thread.wait(60)


def start_auto_delete():
    if st.session_state.get("auto_delete_started"):
        return
    t = threading.Thread(target=auto_delete_loop, daemon=True)
    t.start()
    st.session_state["auto_delete_started"] = True


# ============================================================
# PART 8 - STREAMLIT UI
# Everything below is the web dashboard
# All pages: Dashboard, Scan, History, Stats
# ============================================================

def apply_dashboard_style():
    st.markdown("""
        <style>
            :root {
                --bg: #f7f9fc;
                --panel: #ffffff;
                --line: #d9e2ef;
                --text: #18212f;
                --muted: #657386;
                --danger: #c73637;
                --danger-soft: #fff0f0;
                --success: #16724f;
                --success-soft: #edf8f2;
                --warning: #946200;
                --warning-soft: #fff7e6;
                --info: #1d5f9f;
                --info-soft: #eef6ff;
            }

            .stApp {
                background: var(--bg);
                color: var(--text);
            }

            section[data-testid="stSidebar"] {
                background: #101828;
                border-right: 1px solid #1f2a3d;
            }

            section[data-testid="stSidebar"] * {
                color: #eef3fb !important;
            }

            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2.5rem;
                max-width: 1320px;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            div[data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 16px 18px;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            }

            div[data-testid="stMetricLabel"] {
                color: var(--muted);
                font-size: 0.82rem;
            }

            div[data-testid="stMetricValue"] {
                color: var(--text);
                font-size: 1.75rem;
                font-weight: 750;
            }

            .hero-panel {
                background: linear-gradient(135deg, #10233f 0%, #164a5f 58%, #116048 100%);
                border-radius: 8px;
                padding: 26px 28px;
                color: #ffffff;
                margin-bottom: 18px;
                box-shadow: 0 10px 28px rgba(16, 24, 40, 0.14);
            }

            .hero-panel h1 {
                color: #ffffff;
                font-size: 2rem;
                margin: 0 0 8px 0;
            }

            .hero-panel p {
                color: #dbe8f5;
                margin: 0;
                max-width: 850px;
                line-height: 1.55;
            }

            .section-title {
                margin: 24px 0 10px 0;
                font-size: 1.05rem;
                font-weight: 750;
                color: var(--text);
            }

            .subtle {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .status-panel {
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 16px 18px;
                background: var(--panel);
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            }

            .status-danger {
                border-color: #f0b9ba;
                background: var(--danger-soft);
            }

            .status-safe {
                border-color: #b9e3ce;
                background: var(--success-soft);
            }

            .status-title {
                font-weight: 800;
                font-size: 1.05rem;
                margin-bottom: 8px;
            }

            .kv {
                display: grid;
                grid-template-columns: 140px 1fr;
                gap: 5px 14px;
                font-size: 0.92rem;
                line-height: 1.45;
            }

            .kv span:nth-child(odd) {
                color: var(--muted);
            }

            .badge {
                display: inline-block;
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 4px 10px;
                margin: 2px 5px 6px 0;
                background: #ffffff;
                font-size: 0.8rem;
                color: var(--text);
            }

            .badge-danger {
                color: var(--danger);
                border-color: #f0b9ba;
                background: var(--danger-soft);
            }

            .badge-safe {
                color: var(--success);
                border-color: #b9e3ce;
                background: var(--success-soft);
            }

            .dataframe th {
                background: #f0f4f9;
            }
        </style>
    """, unsafe_allow_html=True)


def render_app_header():
    st.markdown("""
        <div class="hero-panel">
            <h1>AI FraudShield Operations Dashboard</h1>
            <p>
                Real-time Email and SMS fraud detection AI Intelligent System
            </p>
        </div>
    """, unsafe_allow_html=True)


def section_title(title, caption=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="subtle">{caption}</div>', unsafe_allow_html=True)


def render_status_panel(result, sender, subject, msg_type, body):
    is_spam = result["is_spam"]
    panel_class = "status-danger" if is_spam else "status-safe"
    title = "Fraud detected and blocked" if is_spam else "Message cleared"
    badge_class = "badge-danger" if is_spam else "badge-safe"
    safe_sender = html.escape(str(sender) or "Unknown")
    safe_subject = html.escape(str(subject) or "SMS / No subject")
    safe_msg_type = html.escape(str(msg_type))
    safe_label = html.escape(str(result["label"]))
    safe_risk = html.escape(str(result["risk"]))
    safe_scam_type = html.escape(str(result["scam_type"]))
    safe_indicators = html.escape(str(result["indicators"]))
    body_preview = html.escape(str(body))[:420]

    st.markdown(f"""
        <div class="status-panel {panel_class}">
            <div class="status-title">{title}</div>
            <span class="badge {badge_class}">{safe_label}</span>
            <span class="badge">{safe_risk} risk</span>
            <span class="badge">{result["probability"]}% fraud probability</span>
            <div class="kv" style="margin-top: 10px;">
                <span>Sender</span><span>{safe_sender}</span>
                <span>Subject</span><span>{safe_subject}</span>
                <span>Type</span><span>{safe_msg_type}</span>
                <span>Scam type</span><span>{safe_scam_type}</span>
                <span>Signals</span><span>{safe_indicators}</span>
                <span>Model score</span><span>{result["model_probability"]}%</span>
                <span>Message</span><span>{body_preview}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.progress(int(result["probability"]))


def history_rows_to_frame(rows):
    data = []
    for row in rows:
        data.append({
            "Time": str(row[8])[:-3],
            "Label": row[5],
            "Risk": row[7],
            "Scam Type": row[10] or "None",
            "Probability": str(row[6]) + "%",
            "Type": row[4],
            "Sender": row[1],
            "Subject": row[2] or "SMS",
            "Message": str(row[3])[:120],
        })
    return pd.DataFrame(data)

def show_dashboard(model):
    section_title("Command Center", "Monitor fraud volume, detection mix, and recent message decisions from one screen.")
    total, fraud, legit, deleted = get_counts()
    fraud_rate = round((fraud / total) * 100, 1) if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scanned", total)
    col2.metric("Fraud Blocked", fraud, str(fraud_rate) + "% rate")
    col3.metric("Cleared Messages", legit)
    col4.metric("Auto Deleted", deleted)

    section_title("Live Intake", "Simulate incoming Email and SMS traffic to see how the detector responds.")
    col_a, col_b = st.columns([0.95, 2.05])
    with col_a:
        st.markdown('<div class="status-panel">', unsafe_allow_html=True)
        st.write("Queue Controls")
        if st.button("Receive Next Message", use_container_width=True):
            msg    = random.choice(DEMO_MESSAGES)
            result = predict_message(model, msg["body"], msg["sender"], msg["subject"])
            save_to_database(
                msg["sender"], msg["subject"], msg["body"],
                msg["type"], result["label"],
                result["probability"], result["risk"],
                result["scam_type"], result["indicators"]
            )
            st.session_state["last_sim"] = {**msg, **result}
            st.rerun()

        if st.button("Receive 5 Messages", use_container_width=True):
            msgs = random.sample(DEMO_MESSAGES, min(5, len(DEMO_MESSAGES)))
            for msg in msgs:
                result = predict_message(model, msg["body"], msg["sender"], msg["subject"])
                save_to_database(
                    msg["sender"], msg["subject"], msg["body"],
                    msg["type"], result["label"],
                    result["probability"], result["risk"],
                    result["scam_type"], result["indicators"]
                )
            st.success("Scanned 5 messages. Check History to see results.")
            st.rerun()
        st.caption("Messages are sampled from the built-in demo queue.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        if "last_sim" in st.session_state:
            m = st.session_state["last_sim"]
            render_status_panel(m, m["sender"], m["subject"], m["type"], m["body"])
        else:
            st.info("No live message selected yet. Use the intake controls to scan a demo message.")

    section_title("Detection Trends")
    dates, frauds, legits = get_daily_counts()
    chart_data = pd.DataFrame({
        "Fraud"      : frauds,
        "Legitimate" : legits,
    }, index=dates)
    chart_col, mix_col = st.columns([1.55, 1])
    with chart_col:
        st.line_chart(chart_data, height=260)
    with mix_col:
        scam_rows = get_scam_type_counts()
        if scam_rows:
            scam_df = pd.DataFrame(scam_rows, columns=["Scam Type", "Count"])
            st.bar_chart(scam_df.set_index("Scam Type"), height=260)
        else:
            st.info("No fraud categories recorded yet.")

    section_title("Recent Decisions")
    recent_rows = load_history("All", "")[:6]
    if recent_rows:
        st.dataframe(history_rows_to_frame(recent_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No scan history yet.")


def show_scan_page(model):
    section_title("Message Scanner", "Analyze a Email or SMS and store the decision in the case history.")

    st.markdown(
        '<span class="badge">Phishing</span>'
        '<span class="badge">Banking Scam</span>'
        '<span class="badge">OTP Scam</span>'
        '<span class="badge">Malicious URL</span>'
        '<span class="badge">Fake Job Scam</span>'
        '<span class="badge">Loan Offer Spam</span>'
        '<span class="badge">Promotional Ad</span>',
        unsafe_allow_html=True
    )

    section_title("Quick Scenarios")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    if col1.button("Phishing", use_container_width=True):
        st.session_state["scan_body"]    = "Dear user your email account has been suspended. Verify your password at secure-login-update.xyz immediately to restore access."
        st.session_state["scan_sender"]  = "support@mail-security-update.net"
        st.session_state["scan_subject"] = "Urgent account verification"
        st.session_state["scan_type"]    = "Email"

    if col2.button("Bank Scam", use_container_width=True):
        st.session_state["scan_body"]    = "Your SBI KYC is incomplete. Update Aadhaar, PAN, card number and CVV within 24 hours or account will be blocked."
        st.session_state["scan_sender"]  = "kyc-sbi@sbi-kyc-update.net"
        st.session_state["scan_subject"] = "SBI KYC Update Required"
        st.session_state["scan_type"]    = "Email"

    if col3.button("OTP Scam", use_container_width=True):
        st.session_state["scan_body"]    = "Bank support here. Share the OTP you received to cancel an unauthorized transfer from your account immediately."
        st.session_state["scan_sender"]  = "91-88000-22334"
        st.session_state["scan_subject"] = "SMS"
        st.session_state["scan_type"]    = "SMS"

    if col4.button("Bad URL", use_container_width=True):
        st.session_state["scan_body"]    = "Final warning. Login at http://secure-bank-login.xyz/verify-password now to prevent wallet suspension."
        st.session_state["scan_sender"]  = "alerts@secure-bank-login.xyz"
        st.session_state["scan_subject"] = "Wallet suspended"
        st.session_state["scan_type"]    = "Email"

    if col5.button("Fake Job", use_container_width=True):
        st.session_state["scan_body"]    = "You are selected for a work from home job. Earn 5000 daily. Pay registration fee on WhatsApp to start today."
        st.session_state["scan_sender"]  = "hr@fastcareer-offer.com"
        st.session_state["scan_subject"] = "Immediate job selection"
        st.session_state["scan_type"]    = "Email"

    if col6.button("Loan Offer", use_container_width=True):
        st.session_state["scan_body"]    = "Pre approved personal loan up to 5 lakh with low EMI and no documents required. Apply now for quick disbursal."
        st.session_state["scan_sender"]  = "LOANEX"
        st.session_state["scan_subject"] = "SMS"
        st.session_state["scan_type"]    = "SMS"

    if col7.button("Ad Offer", use_container_width=True):
        st.session_state["scan_body"]    = "Exclusive sale today. Flat 70 percent discount, cashback coupon and free gift. Buy now before the offer expires."
        st.session_state["scan_sender"]  = "SHOPAD"
        st.session_state["scan_subject"] = "Mega Sale"
        st.session_state["scan_type"]    = "Email"

    section_title("Scan Input")
    input_col, result_col = st.columns([1.05, 0.95])

    with input_col:
        msg_type = st.selectbox(
            "Message Type",
            ["Email", "SMS"],
            index=["Email", "SMS"].index(st.session_state.get("scan_type", "Email"))
        )
        sender = st.text_input(
            "From / Sender",
            value=st.session_state.get("scan_sender", ""),
            placeholder="example@domain.com or phone number"
        )
        subject = ""
        if msg_type == "Email":
            subject = st.text_input(
                "Subject",
                value=st.session_state.get("scan_subject", ""),
                placeholder="Email subject line"
            )
        body = st.text_area(
            "Message Body",
            value=st.session_state.get("scan_body", ""),
            placeholder="Paste the full message text here...",
            height=220
        )
        scan_clicked = st.button("Scan Message", type="primary", use_container_width=True)

    with result_col:
        st.markdown('<div class="status-panel">', unsafe_allow_html=True)
        st.write("Decision Preview")
        st.caption("Scan output appears here and is written to the message history.")
        st.markdown("</div>", unsafe_allow_html=True)

    if scan_clicked:
        if not body.strip():
            st.warning("Please enter a message to scan.")
        else:
            result = predict_message(model, body, sender, subject)
            save_to_database(
                sender, subject, body, msg_type,
                result["label"], result["probability"], result["risk"],
                result["scam_type"], result["indicators"]
            )
            with result_col:
                render_status_panel(result, sender, subject, msg_type, body)
                if result["is_spam"]:
                    st.warning(
                        "This message has been blocked, saved to history, and scheduled for auto deletion after 24 hours."
                    )
                else:
                    st.success("This message has been saved as legitimate.")

            for key in ["scan_body", "scan_sender", "scan_subject", "scan_type"]:
                if key in st.session_state:
                    del st.session_state[key]


def show_history_page():
    section_title("Message History", "Review scanned messages, fraud categories, confidence levels, and retention timing.")

    col1, col2, col3 = st.columns([1, 1, 2])
    filter_type = col1.selectbox("Status", ["All", "Fraud Only", "Legitimate Only"])
    scam_filter = col2.selectbox(
        "Scam Type",
        [
            "All", "Phishing Email", "Banking Scam", "OTP Scam", "Malicious URL",
            "Fake Job Scam", "Loan Offer Spam", "Promotional Ad", "General Spam"
        ]
    )
    search_text = col3.text_input("Search", placeholder="Sender, subject, message, signal, or scam type")

    total, fraud, legit, deleted = get_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visible Records", total)
    m2.metric("Fraud Records", fraud)
    m3.metric("Legitimate Records", legit)
    m4.metric("Deleted Records", deleted)

    rows = load_history(filter_type, search_text)
    if scam_filter != "All":
        rows = [row for row in rows if (row[10] or "None") == scam_filter]

    if not rows:
        st.info("No messages match the selected filters.")
        return

    st.dataframe(history_rows_to_frame(rows), use_container_width=True, hide_index=True)

    section_title("Record Details")
    for row in rows:
        sender    = row[1]
        subject   = row[2]
        body      = row[3]
        msg_type  = row[4]
        label     = row[5]
        prob      = row[6]
        risk      = row[7]
        scanned   = row[8]
        delete_at = row[9]
        scam_type = row[10] or "None"
        indicators = row[11] or ""

        if label == "FRAUD":
            heading = "Blocked | " + str(msg_type) + " | " + str(scam_type) + " | " + str(prob) + "% | " + str(scanned)[:-3]
            with st.expander(
                heading
            ):
                st.markdown(
                    '<span class="badge badge-danger">FRAUD</span>'
                    f'<span class="badge">{risk} risk</span>'
                    f'<span class="badge">{scam_type}</span>',
                    unsafe_allow_html=True
                )
                c1, c2 = st.columns(2)
                c1.write("From: " + str(sender))
                c1.write("Subject: " + str(subject))
                c2.write("Fraud Probability: " + str(prob) + "%")
                c2.write("Detected Signals: " + str(indicators))
                st.progress(int(prob))
                if delete_at:
                    st.write("Auto deletes at: " + str(delete_at))
                st.text_area("Message", value=str(body), height=120, disabled=True, key="hist_body_" + str(row[0]))
        else:
            heading = "Cleared | " + str(msg_type) + " | " + str(prob) + "% | " + str(scanned)[:-3]
            with st.expander(
                heading
            ):
                st.markdown('<span class="badge badge-safe">LEGITIMATE</span>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.write("From: " + str(sender))
                c1.write("Subject: " + str(subject))
                c2.write("Fraud Probability: " + str(prob) + "%")
                st.progress(int(prob))
                st.text_area("Message", value=str(body), height=120, disabled=True, key="hist_body_" + str(row[0]))


def show_stats_page(model_info):
    section_title("Analytics", "Inspect model performance, fraud patterns, and detection activity over time.")

    total, fraud, legit, deleted = get_counts()
    fraud_rate = round((fraud / total) * 100, 1) if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scanned", total)
    col2.metric("Fraud Rate", str(fraud_rate) + "%")
    col3.metric("Safe Messages", legit)
    col4.metric("Auto Deleted", deleted)

    model_col, matrix_col = st.columns([1, 1])
    if model_info:
        with model_col:
            section_title("Model Profile")
            st.metric("Accuracy", str(model_info["accuracy"]) + "%")
            st.markdown(
                '<div class="status-panel">'
                '<div class="kv">'
                f'<span>Algorithm</span><span>TF-IDF + {str(model_info.get("algorithm", "RandomForest/XGBoost"))}</span>'
                f'<span>Features</span><span>{str(model_info.get("feature_engineering", "word and character NLP features"))}</span>'
                f'<span>Training Data</span><span>{str(model_info["total_data"])} messages</span>'
                f'<span>Spam Samples</span><span>{str(model_info["spam_count"])}</span>'
                f'<span>Ham Samples</span><span>{str(model_info["ham_count"])}</span>'
                f'<span>Scam Examples</span><span>{str(model_info.get("extra_training_examples", 0))}</span>'
                f'<span>Trained At</span><span>{str(model_info["trained_at"])}</span>'
                f'<span>Model Threshold</span><span>{str(int(MODEL_SPAM_THRESHOLD * 100))}%</span>'
                f'<span>Topic Threshold</span><span>{str(int(SPAM_THRESHOLD * 100))}%</span>'
                '</div></div>',
                unsafe_allow_html=True
            )

        with matrix_col:
            section_title("Confusion Matrix")
            cm = model_info["cm"]
            cm_df = pd.DataFrame(
                cm,
                index   = ["Actual Legitimate", "Actual Spam"],
                columns = ["Predicted Legitimate", "Predicted Spam"]
            )
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Model loaded from saved file. Retrain to show full accuracy and matrix details.")

    section_title("Detection Charts")
    dates, frauds, legits = get_daily_counts()
    chart_data = pd.DataFrame({
        "Fraud"      : frauds,
        "Legitimate" : legits,
    }, index=dates)
    trend_col, scam_col = st.columns([1.45, 1])
    with trend_col:
        st.bar_chart(chart_data, height=280)
    with scam_col:
        scam_rows = get_scam_type_counts()
        if scam_rows:
            scam_df = pd.DataFrame(scam_rows, columns=["Scam Type", "Count"])
            st.bar_chart(scam_df.set_index("Scam Type"), height=280)
        else:
            st.info("No fraud categories recorded yet.")

    if model_info:
        section_title("Classification Report")
        st.text(model_info["report"])


def show_retrain_page():
    section_title("Model Training", "Retrain the detector from spam.csv plus the built-in scam examples.")

    if not os.path.exists(DATASET_FILE):
        st.error(
            "spam.csv not found.\n"
            "Please put spam.csv in the same folder as app.py and restart."
        )
        return

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown(
            '<div class="status-panel">'
            '<div class="kv">'
            f'<span>Dataset</span><span>{DATASET_FILE}</span>'
            f'<span>Model File</span><span>{MODEL_FILE}</span>'
            f'<span>Primary Model</span><span>XGBoost when installed</span>'
            f'<span>Fallback Model</span><span>RandomForest</span>'
            f'<span>Extra Scam Examples</span><span>{len(TRAINING_EXAMPLES)}</span>'
            '</div></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.info("Retraining replaces the saved model file and refreshes model statistics in this session.")

    if st.button("Retrain Model Now", type="primary", use_container_width=True):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        model, info = train_model()
        st.session_state["model"]      = model
        st.session_state["model_info"] = info
        st.success(
            "Model retrained successfully.\n"
            "New accuracy: " + str(info["accuracy"]) + "%"
        )
        st.rerun()


# ============================================================
# PART 9 - MAIN FUNCTION - THIS IS WHERE THE APP STARTS
# ============================================================

def main():
    # page config
    st.set_page_config(
        page_title = "FraudShield",
        page_icon  = "F",
        layout     = "wide",
    )

    apply_dashboard_style()
    render_app_header()

    # setup database on first run
    setup_database()

    # start auto delete background thread
    start_auto_delete()

    # load or train model - store in session state so it does not retrain on every click
    if "model" not in st.session_state:
        model, info = get_model()
        st.session_state["model"]      = model
        st.session_state["model_info"] = info

    model      = st.session_state["model"]
    model_info = st.session_state.get("model_info", None)

    # sidebar navigation
    st.sidebar.title("FraudShield")
    st.sidebar.caption("Email and SMS fraud detection")
    page = st.sidebar.radio(
        "Workspace",
        ["Dashboard", "Scan Message", "Message History", "Statistics", "Retrain Model"]
    )

    # show counts in sidebar
    total, fraud, legit, deleted = get_counts()
    st.sidebar.write("---")
    st.sidebar.write("System Snapshot")
    st.sidebar.write("Total Scanned: " + str(total))
    st.sidebar.write("Fraud Blocked: " + str(fraud))
    st.sidebar.write("Safe Messages: " + str(legit))
    st.sidebar.write("Auto Deleted:  " + str(deleted))
    st.sidebar.write("---")
    st.sidebar.write("Detection Topics")
    st.sidebar.write("Phishing Emails")
    st.sidebar.write("Banking Scams")
    st.sidebar.write("OTP Scams")
    st.sidebar.write("Malicious URLs")
    st.sidebar.write("Fake Job Scams")
    st.sidebar.write("Loan Offers")
    st.sidebar.write("Promotional Ads")

    # show which page to display
    if page == "Dashboard":
        show_dashboard(model)

    elif page == "Scan Message":
        show_scan_page(model)

    elif page == "Message History":
        show_history_page()

    elif page == "Statistics":
        show_stats_page(model_info)

    elif page == "Retrain Model":
        show_retrain_page()


# this line runs the app
if __name__ == "__main__":
    main()
