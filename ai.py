import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Evolution of AI",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #020617;
    color: white;
}

.stApp {
    background-color: #020617;
}

h1, h2, h3 {
    color: #38bdf8;
}

div[data-testid="stMetricValue"] {
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET
# =========================================================

dataset_path = "ai_evolution_dataset.csv"

if not os.path.exists(dataset_path):

    sample_data = pd.DataFrame({

        "Year": [1950, 1960, 1980, 1990, 2005, 2015, 2020, 2024],

        "Learning": [0, 0, 1, 1, 1, 1, 1, 1],

        "Automation": [1, 1, 1, 1, 1, 1, 1, 1],

        "Creativity": [0, 0, 0, 0, 0, 0, 1, 1],

        "Generation": [

            "Rule-Based AI",

            "Rule-Based AI",

            "Machine Learning",

            "Machine Learning",

            "Deep Learning",

            "Deep Learning",

            "Generative AI",

            "Generative AI"
        ]
    })

    sample_data.to_csv(dataset_path, index=False)

# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(dataset_path)

X = data.drop("Generation", axis=1)

y = data["Generation"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# =========================================================
# TRAIN MODEL
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# =========================================================
# AI DETAILS
# =========================================================

ai_generations = {

    "Rule-Based AI":
"""
RULE-BASED AI (1950s)

✔ Fixed rules
✔ No learning
✔ Human coded logic

Examples:
• Chess Programs
• Calculators
• Basic Automation
""",

    "Machine Learning":
"""
MACHINE LEARNING (1980s)

✔ Learns from Data
✔ Improves Over Time
✔ Predictive Systems

Examples:
• Recommendations
• Spam Filters
• Fraud Detection
""",

    "Deep Learning":
"""
DEEP LEARNING (2000s)

✔ Neural Networks
✔ Image Recognition
✔ Speech Processing

Examples:
• Face Recognition
• Self Driving Cars
• Medical AI
""",

    "Generative AI":
"""
GENERATIVE AI (2020s)

✔ Creates Content
✔ Generates Text & Images
✔ Human Like Intelligence

Examples:
• ChatGPT
• AI Art
• AI Video Generation
"""
}

# =========================================================
# TITLE
# =========================================================

st.title("EVOLUTION OF ARTIFICIAL INTELLIGENCE")

# =========================================================
# LAYOUT
# =========================================================

left, right = st.columns([1, 2])

# =========================================================
# LEFT PANEL
# =========================================================

with left:

    st.header("INTRODUCTION TO AI")

    st.write("""
Artificial Intelligence (AI) is the simulation
of human intelligence in machines.

### AI Evolution

- 1950 → Rule-Based AI
- 1980 → Machine Learning
- 2000 → Deep Learning
- 2020 → Generative AI

### Modern AI Can

- Learn
- Predict
- Create
- Automate
""")

    st.subheader("MODEL INFO")

    st.success("Decision Tree Classifier")

    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

# =========================================================
# RIGHT PANEL
# =========================================================

with right:

    st.header("AI EVOLUTION PREDICTOR")

    year = st.number_input(
        "Enter Year",
        min_value=1950,
        max_value=2100,
        value=2024
    )

    learning = st.checkbox("Learning Capability")

    automation = st.checkbox("Automation")

    creativity = st.checkbox("Creative AI")

    if st.button("PREDICT AI GENERATION"):

        sample = np.array([[
            year,
            int(learning),
            int(automation),
            int(creativity)
        ]])

        prediction = model.predict(sample)

        result = encoder.inverse_transform(
            prediction
        )[0]

        probability = np.max(
            model.predict_proba(sample)
        ) * 100

        st.success(f"Predicted Generation: {result}")

        st.info(
            f"Confidence: {probability:.2f}%"
        )

        st.code(ai_generations[result])

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("Built with Streamlit + Machine Learning")