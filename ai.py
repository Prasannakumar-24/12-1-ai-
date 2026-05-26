import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================================================
# WINDOW
# =========================================================

root = tk.Tk()

root.title("Evolution of Artificial Intelligence")

root.geometry("1400x800")

root.configure(bg="#020617")

# =========================================================
# COLORS
# =========================================================

BG = "#020617"
CARD = "#0f172a"
BLUE = "#38bdf8"
GREEN = "#22c55e"
YELLOW = "#facc15"
RED = "#ef4444"
WHITE = "white"

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
# LOAD DATASET
# =========================================================

try:

    data = pd.read_csv(dataset_path)

except Exception as e:

    messagebox.showerror(
        "Dataset Error",
        f"Failed to load dataset\n\n{e}"
    )

    sys.exit()

# =========================================================
# MODEL
# =========================================================

X = data.drop("Generation", axis=1)

y = data["Generation"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

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

title_label = tk.Label(
    root,
    text="EVOLUTION OF ARTIFICIAL INTELLIGENCE",
    font=("Arial", 30, "bold"),
    bg=BG,
    fg=BLUE
)

title_label.pack(pady=20)

# =========================================================
# TITLE ANIMATION
# =========================================================

title_colors = [BLUE, GREEN, YELLOW, RED]

title_index = 0

def animate_title():

    global title_index

    title_label.config(
        fg=title_colors[title_index]
    )

    title_index += 1

    if title_index >= len(title_colors):

        title_index = 0

    root.after(500, animate_title)

animate_title()

# =========================================================
# MAIN FRAME
# =========================================================

main_frame = tk.Frame(root, bg=BG)

main_frame.pack(fill="both", expand=True)

# =========================================================
# LEFT PANEL
# =========================================================

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

# =========================================================
# INTRODUCTION
# =========================================================

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

# =========================================================
# MODEL INFO
# =========================================================

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

# =========================================================
# RIGHT PANEL
# =========================================================

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

# =========================================================
# INPUT FRAME
# =========================================================

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

# =========================================================
# YEAR INPUT
# =========================================================

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

# =========================================================
# VARIABLES
# =========================================================

learning_var = tk.IntVar()

automation_var = tk.IntVar()

creativity_var = tk.IntVar()

# =========================================================
# CHECKBOXES
# =========================================================

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

# =========================================================
# BUTTONS
# =========================================================

button_frame = tk.Frame(
    input_frame,
    bg=CARD
)

button_frame.pack(pady=20)

# =========================================================
# RESULT FRAME
# =========================================================

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
    bg="#111827",
    fg=GREEN
)

result_label.pack(pady=20)

confidence_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 18),
    bg="#111827",
    fg=BLUE
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

# =========================================================
# TYPING EFFECT
# =========================================================

typing_text = ""

typing_index = 0

def typing_effect():

    global typing_index

    if typing_index <= len(typing_text):

        output_label.config(
            text=typing_text[:typing_index]
        )

        typing_index += 1

        root.after(15, typing_effect)

# =========================================================
# PREDICTION
# =========================================================

def predict_generation():

    global typing_text
    global typing_index

    try:

        year = int(year_entry.get())

        sample = np.array([[
            year,
            learning_var.get(),
            automation_var.get(),
            creativity_var.get()
        ]])

        prediction = model.predict(sample)

        result = encoder.inverse_transform(
            prediction
        )[0]

        probability = np.max(
            model.predict_proba(sample)
        ) * 100

        result_label.config(text=result)

        confidence_label.config(
            text=f"Confidence: {probability:.2f}%"
        )

        typing_text = ai_generations[result]

        typing_index = 0

        typing_effect()

    except Exception as e:

        messagebox.showerror(
            "Input Error",
            str(e)
        )

# =========================================================
# RESET
# =========================================================

def reset_all():

    year_entry.delete(0, tk.END)

    learning_var.set(0)

    automation_var.set(0)

    creativity_var.set(0)

    result_label.config(text="")

    confidence_label.config(text="")

    output_label.config(text="")

# =========================================================
# BUTTONS
# =========================================================

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

# =========================================================
# PROGRESS BAR
# =========================================================

progress = ttk.Progressbar(
    root,
    orient="horizontal",
    mode="determinate",
    length=1200
)

progress.pack(pady=10)

progress_value = 0

direction = 1

def animate_progress():

    global progress_value
    global direction

    progress_value += direction

    if progress_value >= 100:

        direction = -1

    elif progress_value <= 0:

        direction = 1

    progress["value"] = progress_value

    root.after(20, animate_progress)

animate_progress()

# =========================================================
# SAFE CLOSE
# =========================================================

def safe_close():

    try:

        root.quit()

        root.destroy()

    except:

        pass

    os._exit(0)

root.protocol("WM_DELETE_WINDOW", safe_close)

# =========================================================
# MAINLOOP
# =========================================================

try:

    root.mainloop()

except KeyboardInterrupt:

    try:

        safe_close()

    except:

        pass