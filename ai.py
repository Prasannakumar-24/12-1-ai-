import tkinter as tk
from tkinter import ttk, messagebox
import csv
import importlib.util

pandas_spec = importlib.util.find_spec("pandas")
if pandas_spec is not None:
    pd = importlib.import_module("pandas")
else:
    pd = None
import numpy as np
import threading
import time
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# WINDOW
# ----------------------------------------------------

root = tk.Tk()
root.title("Evolution of AI")
root.geometry("1400x800")
root.config(bg="#020617")

# ----------------------------------------------------
# COLORS
# ----------------------------------------------------

BG = "#020617"
CARD = "#0f172a"
BLUE = "#38bdf8"
GREEN = "#22c55e"
YELLOW = "#facc15"
RED = "#ef4444"
WHITE = "white"

# ----------------------------------------------------
# DATASET CHECK
# ----------------------------------------------------

dataset_path = "ai_evolution_dataset.csv"

if not os.path.exists(dataset_path):

    messagebox.showerror(
        "Dataset Missing",
        f"{dataset_path} not found!"
    )

    root.destroy()
    raise SystemExit

# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------

if pd is not None:
    data = pd.read_csv(dataset_path)
    X = data.drop("Generation", axis=1)
    y = data["Generation"]
else:
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows or "Generation" not in reader.fieldnames:
        messagebox.showerror(
            "Dataset Error",
            "Invalid dataset format or missing 'Generation' column."
        )
        root.destroy()
        raise SystemExit

    feature_names = [name for name in reader.fieldnames if name != "Generation"]
    X = np.array([
        [float(row[name]) for name in feature_names]
        for row in rows
    ])
    y = np.array([row["Generation"] for row in rows])

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ----------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------
# MACHINE LEARNING MODEL
# ----------------------------------------------------

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# ----------------------------------------------------
# TITLE ANIMATION
# ----------------------------------------------------

def animate_title():

    colors = [BLUE, GREEN, YELLOW, RED]

    while True:

        for color in colors:

            title_label.config(fg=color)

            time.sleep(0.5)

# ----------------------------------------------------
# TYPING EFFECT
# ----------------------------------------------------

def typing_effect(text):

    output_label.config(text="")

    for i in range(len(text)+1):

        output_label.config(text=text[:i])

        root.update()

        time.sleep(0.01)

# ----------------------------------------------------
# AI EVOLUTION DETAILS
# ----------------------------------------------------

ai_generations = {

    "Rule-Based AI":
"""
RULE-BASED AI (1950s)

✔ Fixed rules
✔ No learning
✔ Human coded logic

Examples:
• Chess programs
• Calculators
• Basic automation
""",

    "Machine Learning":
"""
MACHINE LEARNING (1980s)

✔ Learns from data
✔ Improves over time
✔ Predictive systems

Examples:
• Recommendations
• Spam filters
• Fraud detection
""",

    "Deep Learning":
"""
DEEP LEARNING (2000s)

✔ Neural networks
✔ Image recognition
✔ Speech processing

Examples:
• Face recognition
• Self-driving cars
• Medical AI
""",

    "Generative AI":
"""
GENERATIVE AI (2020s)

✔ Creates content
✔ Generates text/images
✔ Human-like intelligence

Examples:
• ChatGPT
• AI Art
• AI Video generation
"""
}

# ----------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------

def predict_generation():

    try:

        year = int(year_entry.get())

        learning = int(learning_var.get())

        automation = int(automation_var.get())

        creativity = int(creativity_var.get())

        sample = np.array([[
            year,
            learning,
            automation,
            creativity
        ]])

        prediction = model.predict(sample)

        result = encoder.inverse_transform(prediction)[0]

        probability = np.max(
            model.predict_proba(sample)
        ) * 100

        result_label.config(
            text=result,
            fg=GREEN
        )

        confidence_label.config(
            text=f"Confidence: {probability:.2f}%",
            fg=BLUE
        )

        threading.Thread(
            target=typing_effect,
            args=(ai_generations[result],),
            daemon=True
        ).start()

    except:

        messagebox.showerror(
            "Input Error",
            "Please enter valid values."
        )

# ----------------------------------------------------
# RESET FUNCTION
# ----------------------------------------------------

def reset_all():

    year_entry.delete(0, tk.END)

    learning_var.set(0)

    automation_var.set(0)

    creativity_var.set(0)

    result_label.config(text="")

    confidence_label.config(text="")

    output_label.config(text="")

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------

title_label = tk.Label(
    root,
    text="EVOLUTION OF ARTIFICIAL INTELLIGENCE",
    font=("Arial", 32, "bold"),
    bg=BG,
    fg=BLUE
)

title_label.pack(pady=20)

threading.Thread(
    target=animate_title,
    daemon=True
).start()

# ----------------------------------------------------
# MAIN FRAME
# ----------------------------------------------------

main_frame = tk.Frame(root, bg=BG)

main_frame.pack(fill="both", expand=True)

# ----------------------------------------------------
# LEFT PANEL
# ----------------------------------------------------

left_panel = tk.Frame(
    main_frame,
    bg=CARD,
    width=400
)

left_panel.pack(
    side="left",
    fill="y",
    padx=20,
    pady=20
)

# ----------------------------------------------------
# INTRODUCTION
# ----------------------------------------------------

tk.Label(
    left_panel,
    text="INTRODUCTION TO AI",
    font=("Arial", 22, "bold"),
    bg=CARD,
    fg=BLUE
).pack(pady=20)

intro_text = """
Artificial Intelligence (AI)
is the simulation of human
intelligence in machines.

AI Evolution:
1950 → Rule-Based AI
1980 → Machine Learning
2000 → Deep Learning
2020 → Generative AI

Modern AI can:
✔ Learn
✔ Predict
✔ Create
✔ Automate
"""

tk.Label(
    left_panel,
    text=intro_text,
    justify="left",
    font=("Arial", 14),
    bg=CARD,
    fg=WHITE
).pack(padx=20)

# ----------------------------------------------------
# MODEL INFO
# ----------------------------------------------------

model_frame = tk.Frame(
    left_panel,
    bg="#1e293b"
)

model_frame.pack(
    fill="x",
    padx=20,
    pady=20
)

tk.Label(
    model_frame,
    text="Machine Learning Model",
    font=("Arial", 16, "bold"),
    bg="#1e293b",
    fg=GREEN
).pack(pady=10)

tk.Label(
    model_frame,
    text="Decision Tree Classifier",
    font=("Arial", 13),
    bg="#1e293b",
    fg=WHITE
).pack()

tk.Label(
    model_frame,
    text=f"Accuracy: {accuracy*100:.2f}%",
    font=("Arial", 14, "bold"),
    bg="#1e293b",
    fg=BLUE
).pack(pady=10)

# ----------------------------------------------------
# RIGHT PANEL
# ----------------------------------------------------

right_panel = tk.Frame(
    main_frame,
    bg=BG
)

right_panel.pack(
    side="right",
    fill="both",
    expand=True,
    padx=20,
    pady=20
)

# ----------------------------------------------------
# INPUT FRAME
# ----------------------------------------------------

input_frame = tk.Frame(
    right_panel,
    bg=CARD
)

input_frame.pack(fill="x")

tk.Label(
    input_frame,
    text="AI EVOLUTION PREDICTOR",
    font=("Arial", 24, "bold"),
    bg=CARD,
    fg=WHITE
).pack(pady=20)

# ----------------------------------------------------
# YEAR INPUT
# ----------------------------------------------------

year_frame = tk.Frame(
    input_frame,
    bg=CARD
)

year_frame.pack(pady=10)

tk.Label(
    year_frame,
    text="Year:",
    font=("Arial", 16),
    bg=CARD,
    fg=WHITE
).grid(row=0, column=0, padx=10)

year_entry = tk.Entry(
    year_frame,
    font=("Arial", 16),
    width=10
)

year_entry.grid(row=0, column=1)

# ----------------------------------------------------
# VARIABLES
# ----------------------------------------------------

learning_var = tk.IntVar()

automation_var = tk.IntVar()

creativity_var = tk.IntVar()

# ----------------------------------------------------
# CHECKBOXES
# ----------------------------------------------------

features = [
    ("Learning Capability", learning_var),
    ("Automation", automation_var),
    ("Creative AI", creativity_var)
]

for text, var in features:

    tk.Checkbutton(
        input_frame,
        text=text,
        variable=var,
        font=("Arial", 15),
        bg=CARD,
        fg=WHITE,
        activebackground=CARD,
        activeforeground=BLUE,
        selectcolor="#334155"
    ).pack(anchor="w", padx=40, pady=10)

# ----------------------------------------------------
# BUTTONS
# ----------------------------------------------------

button_frame = tk.Frame(
    input_frame,
    bg=CARD
)

button_frame.pack(pady=20)

tk.Button(
    button_frame,
    text="PREDICT AI GENERATION",
    command=predict_generation,
    font=("Arial", 14, "bold"),
    bg=GREEN,
    fg="white",
    width=22,
    height=2,
    bd=0
).grid(row=0, column=0, padx=20)

tk.Button(
    button_frame,
    text="RESET",
    command=reset_all,
    font=("Arial", 14, "bold"),
    bg=RED,
    fg="white",
    width=15,
    height=2,
    bd=0
).grid(row=0, column=1, padx=20)

# ----------------------------------------------------
# RESULT FRAME
# ----------------------------------------------------

result_frame = tk.Frame(
    right_panel,
    bg="#111827"
)

result_frame.pack(
    fill="both",
    expand=True,
    pady=20
)

result_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 28, "bold"),
    bg="#111827"
)

result_label.pack(pady=20)

confidence_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 18),
    bg="#111827",
    fg=WHITE
)

confidence_label.pack()

output_label = tk.Label(
    result_frame,
    text="",
    justify="left",
    font=("Arial", 15),
    bg="#111827",
    fg=WHITE
)

output_label.pack(pady=30)

# ----------------------------------------------------
# PROGRESS BAR
# ----------------------------------------------------

progress = ttk.Progressbar(
    root,
    orient="horizontal",
    mode="determinate",
    length=1200
)

progress.pack(pady=10)

def animate_progress():

    while True:

        for i in range(101):

            progress["value"] = i

            root.update()

            time.sleep(0.02)

        for i in range(100, -1, -1):

            progress["value"] = i

            root.update()

            time.sleep(0.02)

threading.Thread(
    target=animate_progress,
    daemon=True
).start()

# ----------------------------------------------------
# RUN
# ----------------------------------------------------

root.mainloop()