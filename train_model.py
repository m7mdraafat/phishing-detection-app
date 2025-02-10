import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import joblib
from urllib.parse import urlparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("data/url_dataset.csv")

# Feature Engineering
def extract_features(url):
    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse("http://" + url)  # Handle invalid URLs

    # Domain features
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # Initialize feature dictionary
    features = {
        # Basic URL characteristics
        "url_length": len(url),
        "domain_length": len(domain),
        "num_dots": domain.count("."),
        "num_hyphens": domain.count("-"),
        "num_underscore": domain.count("_"),
        "num_slash": url.count("/"),
        "num_questionmark": url.count("?"),
        "num_equal": url.count("="),
        "num_ampersand": url.count("&"),
        "num_at": url.count("@"),
        "has_https": 1 if parsed.scheme == "https" else 0,
        # Suspicious patterns
        "ip_in_domain": 1 if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain) else 0,
        "is_shortened": 1 if any(x in domain for x in ["bit.ly", "goo.gl", "tinyurl", "ow.ly"]) else 0,
        "has_subdomain": 1 if domain.count(".") > 1 else 0,
        "is_redirect": 1 if "//" in url[7:] else 0,  # Check after http://
        # Path and query features
        "num_subdirs": path.count("/"),
        "has_port": 1 if ":" in domain else 0,
        "num_params": len(query.split("&")) if query else 0,
        "has_php_ext": 1 if ".php" in path else 0,
        "has_html_ext": 1 if ".html" in path else 0,
        # Sensitive keywords
        "has_login": 1 if re.search(r"login|signin|auth|secure", url, re.I) else 0,
        "has_account": 1 if re.search(r"account|verify|update|confirm", url, re.I) else 0,
        "has_bank_keywords": 1 if re.search(r"bank|paypal|ebay|credit", url, re.I) else 0,
        # Additional security features
        "has_anchor": 1 if "#" in url else 0,
        "has_fake_https": 1 if "https" in path.lower() else 0,
        "url_ratio_digits": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
    }
    return features

# Create feature matrix
X = pd.DataFrame([extract_features(url) for url in df['url']])

# Convert to numerical labels
y = df["type"].map({'phishing': 1, 'legitimate': 0}).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Build final pipeline
model = Pipeline([
    ('clf', RandomForestClassifier(n_estimators=100))  # Classifier
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Phishing'], 
            yticklabels=['Legitimate', 'Phishing'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Feature Importance
feature_importances = model.named_steps['clf'].feature_importances_
feature_names = X.columns

# Sort feature importances
indices = np.argsort(feature_importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, "model/phishing_model.pkl")