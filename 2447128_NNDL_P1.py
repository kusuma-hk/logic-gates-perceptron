
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Activation function
# ----------------------------
def step_function(x):
    return np.where(x >= 0, 1, 0)

# ----------------------------
# Perceptron Training (detailed)
# ----------------------------
def train_perceptron_detailed(X, y, lr=0.1, epochs=10, random_init=True):
    n_samples, n_features = X.shape
    if random_init:
        weights = np.random.randn(n_features)
        bias = np.random.randn()
    else:
        weights = np.zeros(n_features)
        bias = 0.0

    history = []
    training_details = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(n_samples):
            linear_output = np.dot(X[i], weights) + bias
            y_pred = step_function(linear_output)
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error
            total_error += abs(error)
            # Record detailed training info
            training_details.append({
                "Epoch": epoch+1,
                "Input": X[i],
                "Linear Output": round(linear_output, 3),
                "Predicted": int(y_pred),
                "Target": y[i],
                "Error": error,
                "Weights": weights.copy(),
                "Bias": round(bias,3)
            })
        history.append((epoch, weights.copy(), bias, total_error))
    return weights, bias, history, training_details

# ----------------------------
# Datasets for logic gates
# ----------------------------
def get_dataset(gate):
    if gate == "AND":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,0,0,1])
    elif gate == "OR":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,1,1,1])
    elif gate == "AND-NOT":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,0,1,0])
    elif gate == "XOR":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([0,1,1,0])
    return X, y

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Perceptron Logic Gates", page_icon="🤖", layout="wide")
st.title("🔌 Single Layer Perceptron for Logic Gates (Detailed Training)")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
gate = st.sidebar.selectbox("Choose Logic Gate", ["AND", "OR", "AND-NOT", "XOR"])
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 1, 20, 10, 1)
random_init = st.sidebar.radio("Weights Initialization", ["Random", "Zero"]) == "Random"

# Load dataset
X, y = get_dataset(gate)
st.subheader(f"Truth Table: {gate} Gate")
truth_table = pd.DataFrame(X, columns=["Input1","Input2"])
truth_table["Target Output"] = y
st.table(truth_table)

# Train perceptron
weights, bias, history, details = train_perceptron_detailed(X, y, lr=learning_rate, epochs=epochs, random_init=random_init)

# ----------------------------
# Show detailed training process
# ----------------------------
st.subheader("📝 Detailed Training Process")
details_df = pd.DataFrame(details)
st.dataframe(details_df)

# ----------------------------
# Test results
# ----------------------------
st.subheader("🧪 Perceptron Predictions")
outputs = []
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    y_pred = step_function(linear_output)
    outputs.append(int(y_pred))

results_df = pd.DataFrame(X, columns=["Input1", "Input2"])
results_df["Expected"] = y
results_df["Predicted"] = outputs
st.table(results_df)

# ----------------------------
# Plot error and weights
# ----------------------------
st.subheader("📉 Training Progress Visualizations")
fig, ax = plt.subplots(1,2,figsize=(12,4))

# Errors per epoch
errors = [h[3] for h in history]
ax[0].plot(range(1,len(errors)+1), errors, marker="o")
ax[0].set_title("Total Errors per Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Total Error")

# Weight & bias updates
w1 = [h[1][0] for h in history]
w2 = [h[1][1] for h in history]
b = [h[2] for h in history]
ax[1].plot(range(1,len(w1)+1), w1, label="w1")
ax[1].plot(range(1,len(w2)+1), w2, label="w2")
ax[1].plot(range(1,len(b)+1), b, label="bias")
ax[1].set_title("Weights & Bias per Epoch")
ax[1].legend()

st.pyplot(fig)

# XOR note
if gate == "XOR":
    st.warning("⚠️ XOR is NOT linearly separable. A single-layer perceptron cannot solve it perfectly.")
     