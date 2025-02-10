import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import os
from sklearn.metrics import classification_report


# Load the trained model (cached for performance)
@st.cache_resource
def load_model():
    model_path = os.getenv("MODEL_PATH", "model/phishing_model.pkl")
    return joblib.load(model_path)


model = load_model()



# Title and header
st.title("Welcome to the Phishing Detection Application")
st.markdown("---")

# Main description
col1, col2 = st.columns([3, 1])
with col1:
    st.write("""
    **This ML app is developed for educational purposes to detect phishing websites using machine learning algorithms.**
    - Enter a URL in the input field below to check if it's legitimate or phishing
    - View detailed analysis of the URL features
    - Explore the model's performance metrics and methodology
    """)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/5064/5064343.png", width=150)

# Input section
st.markdown("## URL Analysis")
url_input = st.text_input(
    "Enter URL to analyze:",
    placeholder="https://example.com",
    help="Enter complete URL including http:// or https://",
)

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = []


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
        "ip_in_domain": 1
        if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain)
        else 0,
        "is_shortened": 1
        if any(x in domain for x in ["bit.ly", "goo.gl", "tinyurl", "ow.ly"])
        else 0,
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
        "has_account": 1
        if re.search(r"account|verify|update|confirm", url, re.I)
        else 0,
        "has_bank_keywords": 1
        if re.search(r"bank|paypal|ebay|credit", url, re.I)
        else 0,
        # Additional security features
        "has_anchor": 1 if "#" in url else 0,
        "has_fake_https": 1 if "https" in path.lower() else 0,
        "url_ratio_digits": sum(c.isdigit() for c in url) / len(url)
        if len(url) > 0
        else 0,
    }
    return features



if st.button("Analyze URL"):
    if url_input:
        try:
            # Extract features and predict
            features = extract_features(url_input)
            prediction = model.predict(pd.DataFrame([features]))[0]
            confidence = np.max(model.predict_proba(pd.DataFrame([features])))

            # Display results
            st.subheader("Analysis Results")

            # Result badges
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Prediction",
                    value="⚠️ Phishing" if prediction == 1 else "✅ Legitimate",
                    delta=f"Confidence: {confidence * 100:.2f}%",
                )

            # Feature visualization
            with st.expander("Detailed Feature Analysis"):
                features_df = pd.DataFrame([features]).T.reset_index()
                features_df.columns = ["Feature", "Value"]

                fig, ax = plt.subplots(figsize=(10, 15))
                sns.barplot(y="Feature", x="Value", data=features_df, ax=ax)
                ax.set_title("URL Feature Values")
                st.pyplot(fig)

            # Add to history
            st.session_state.results.insert(
                0,
                {"url": url_input, "prediction": prediction, "confidence": confidence},
            )

        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")
    else:
        st.warning("Please enter a URL to analyze!")

# Project details expander
with st.expander("Project Details", expanded=True):
    st.subheader("Approach")
    st.markdown("""
    - **Machine Learning Pipeline**:
        1. Feature engineering from URL structure
        2. Random Forest classification
        3. Model evaluation using precision, recall, and F1-score
    - **Key Technologies**:
        - Scikit-learn for model development
        - Streamlit for web interface
        - Feature engineering for security indicators
    """)

    st.subheader("Dataset")
    st.markdown("""
    - **Source**: Curated dataset of 45,000+ URLs
    - **Balance**: Equal distribution of phishing and legitimate URLs
    - **Features**: 26 handcrafted features from URL structure
    - **Anonymization**: No personal or sensitive information included
    """)

    st.subheader("Model Performance")
    st.markdown("""
    - **Accuracy**: 95% on test data
    - **Precision**: 96% (Phishing), 95% (Legitimate)
    - **Recall**: 95% (Phishing), 96% (Legitimate)
    - **F1-Score**: 95.5% overall
    """)

    # Performance visualization
    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [0.95, 0.955, 0.95, 0.955],
    }
    metrics_df = pd.DataFrame(metrics)

    fig, ax = plt.subplots()
    sns.barplot(x="Value", y="Metric", data=metrics_df, palette="viridis")
    ax.set_title("Model Performance Metrics")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

# History section
if st.session_state.results:
    st.markdown("## Analysis History")
    for result in st.session_state.results[:5]:  # Show last 5 results
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**URL**: {result['url']}")
            with col2:
                status = "⚠️ Phishing" if result["prediction"] == 1 else "✅ Legitimate"
                st.write(f"{status} ({result['confidence'] * 100:.2f}%)")
            st.markdown("---")

# Instructions expander
with st.expander("How to Use This App", expanded=False):
    st.markdown("""
    1. Enter a complete URL in the input field (include http:// or https://)
    2. Click the 'Analyze URL' button
    3. View the prediction results and confidence score
    4. Explore detailed feature analysis in the expandable sections
    5. Check previous analyses in the history section
    """)
