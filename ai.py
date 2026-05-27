import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI vs Non-AI Detector",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #020617;
    color: white;
}

h1, h2, h3 {
    color: #38bdf8;
}

.block-container {
    padding-top: 2rem;
}

div.stButton > button {
    background-color: #0ea5e9;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #0284c7;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DATASET FILE
# =====================================================

dataset_file = "ai_dataset.csv"

# =====================================================
# CREATE DATASET IF NOT EXISTS
# =====================================================

if not os.path.exists(dataset_file):

    sample_data = pd.DataFrame({

        "Learning": [1,0,1,0,1,0,1,0,1,0],

        "UsesData": [1,0,1,0,1,0,1,0,1,0],

        "DecisionMaking": [1,0,1,0,1,0,1,0,1,0],

        "PatternRecognition": [1,0,1,0,1,0,1,0,1,0],

        "HumanProgramming": [0,1,0,1,0,1,0,1,0,1],

        "Result": [
            "AI",
            "Non-AI",
            "AI",
            "Non-AI",
            "AI",
            "Non-AI",
            "AI",
            "Non-AI",
            "AI",
            "Non-AI"
        ]
    })

    sample_data.to_csv(dataset_file, index=False)

# =====================================================
# LOAD DATA
# =====================================================

try:
    data = pd.read_csv(dataset_file)

except Exception as e:
    st.error(f"Dataset Loading Error: {e}")
    st.stop()

# =====================================================
# FEATURES & TARGET
# =====================================================

X = data.drop("Result", axis=1)

y = data["Result"]

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL TRAINING
# =====================================================

try:

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

except Exception as e:

    st.error(f"Model Training Error: {e}")
    st.stop()

# =====================================================
# TITLE
# =====================================================

st.title("🤖 AI vs NON-AI DETECTOR")

st.markdown("""
Detect whether a software/application is AI-based
or traditional software using Machine Learning.
""")

# =====================================================
# LAYOUT
# =====================================================

left, right = st.columns([1, 2])

# =====================================================
# LEFT PANEL
# =====================================================

with left:

    st.header("📘 WHAT IS AI?")

    st.write("""
Artificial Intelligence (AI) is technology
that enables machines to learn, analyze,
predict, and make smart decisions.

### Examples
- ChatGPT
- Self-driving Cars
- AI Robots
- Face Detection
- Voice Assistants

### AI Systems Can
- Learn from data
- Predict outcomes
- Analyze information
- Make decisions
- Recognize patterns
""")

    st.subheader("📊 MODEL INFO")

    st.success("Decision Tree Classifier")

    st.metric(
        label="Accuracy",
        value=f"{accuracy * 100:.2f}%"
    )

    st.subheader("📁 Dataset Preview")

    st.dataframe(data)

# =====================================================
# RIGHT PANEL
# =====================================================

with right:

    st.header("🧠 AI APPLICATION DETECTOR")

    st.write("Select the features available in the application.")

    learning = st.checkbox("Uses Learning")

    uses_data = st.checkbox("Uses Data")

    decision = st.checkbox("Makes Decisions")

    pattern = st.checkbox("Recognizes Patterns")

    human = st.checkbox("Only Human Programming")

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button("🚀 PREDICT"):

        try:

            sample = np.array([[
                int(learning),
                int(uses_data),
                int(decision),
                int(pattern),
                int(human)
            ]])

            # FIXED FEATURE NAME ISSUE
            sample_df = pd.DataFrame(
                sample,
                columns=X.columns
            )

            result = model.predict(sample_df)

            prediction = encoder.inverse_transform(result)[0]

            confidence = np.max(
                model.predict_proba(sample_df)
            ) * 100

            st.subheader("📌 RESULT")

            if prediction == "AI":

                st.success("✅ AI APPLICATION DETECTED")

                explanation = """
✔ Learns from data

✔ Makes smart decisions

✔ Uses automation

✔ Recognizes patterns

✔ Intelligent behavior detected
"""

            else:

                st.error("❌ NON-AI APPLICATION")

                explanation = """
✘ No learning capability

✘ Mostly manually programmed

✘ Limited automation

✘ No pattern recognition

✘ Traditional software system
"""

            st.info(
                f"Confidence Level: {confidence:.2f}%"
            )

            st.code(explanation)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("Built with Streamlit + Scikit-Learn + Machine Learning")