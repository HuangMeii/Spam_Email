import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from lstm_explainer import LSTMExplainerCore


# =========================
# LOAD MODEL (your trained model)
# =========================
def load_weights(model):
    lstm = model.lstm

    W = lstm.weight_ih_l0.detach().cpu().numpy()
    U = lstm.weight_hh_l0.detach().cpu().numpy()
    b = lstm.bias_ih_l0.detach().cpu().numpy() + lstm.bias_hh_l0.detach().cpu().numpy()

    hidden = lstm.hidden_size

    return W, U, b, hidden


# =========================
# UI
# =========================
st.title("🧠 LSTM Gate Explainer (Spam Email Demo)")

st.write("Chọn 1 email sample và xem LSTM hoạt động theo từng word")

# fake sample input (replace with real embedding)
seq_len = 20
emb_dim = 300

x_seq = np.random.randn(seq_len, emb_dim)

word_list = [f"word_{i}" for i in range(seq_len)]


# slider
t = st.slider("Timestep (word)", 0, seq_len - 1, 0)


# load model weights placeholder
st.warning("Replace this with your trained model")

# fake weights (demo)
W = np.random.randn(4 * 64, emb_dim)
U = np.random.randn(4 * 64, 64)
b = np.random.randn(4 * 64)

explainer = LSTMExplainerCore(W, U, b, hidden_size=64)


# run step-by-step
h = np.zeros(64)
c = np.zeros(64)

history = []

for i in range(t + 1):
    f, i_g, c_tilde, o, h, c = explainer.step(x_seq[i], h, c)

    history.append({
        "word": word_list[i],
        "f": np.mean(f),
        "i": np.mean(i_g),
        "c": np.mean(c_tilde),
        "o": np.mean(o)
    })


st.subheader(f"Word: {word_list[t]}")

col1, col2 = st.columns(2)

with col1:
    st.metric("Forget gate f_t", f"{history[-1]['f']:.3f}")
    st.metric("Input gate i_t", f"{history[-1]['i']:.3f}")

with col2:
    st.metric("Candidate c~_t", f"{history[-1]['c']:.3f}")
    st.metric("Output gate o_t", f"{history[-1]['o']:.3f}")


# =========================
# PLOT
# =========================
st.subheader("📊 Gate evolution")

fig, ax = plt.subplots()

ax.plot([h["f"] for h in history], label="forget")
ax.plot([h["i"] for h in history], label="input")
ax.plot([h["c"] for h in history], label="candidate")
ax.plot([h["o"] for h in history], label="output")

ax.legend()

st.pyplot(fig)