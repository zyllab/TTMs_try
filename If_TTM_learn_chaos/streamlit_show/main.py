import streamlit as st
import pandas as pd
import numpy as np
import nolds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import pickle
import io
from sklearn.metrics import mutual_info_score

# --- Utils ---


@st.cache_data
def load_data_from_hf(url: str):
    response = requests.get(url)
    data = pickle.load(io.BytesIO(response.content))
    return data

@st.cache_data
def load_les_data():
    les_df = df = pd.read_pickle("les_org.pkl")
    return les_df


def delay_embedding_3d(series, delay=1):
    return np.column_stack([
        series[:-2 * delay],
        series[delay:-delay],
        series[2 * delay:]
    ])

def mf_value(x):
    n = len(x)
    fft_vals = np.fft.rfft(x,n)
    psd = np.abs(fft_vals)**2
    freqs = np.fft.rfftfreq(n)
    # Skip DC component at index 0
    mean_freq = np.sum(freqs[1:] * psd[1:]) / np.sum(psd[1:])
    return mean_freq

def compute_ami_lag(x, max_lag=100, bins=32):
    """
    Estimate optimal embedding lag via average mutual information (AMI).
    Returns list of AMI values and the first local minimum lag.
    """
    # Discretize data into bins
    hist, bin_edges = np.histogram(x, bins=bins)
    digitized = np.digitize(x, bin_edges[:-1])
    
    ami = []
    for lag in range(1, max_lag):
        x1 = digitized[:-lag]
        x2 = digitized[lag:]
        mi = mutual_info_score(x1, x2)
        ami.append(mi)

    # Find the first local minimum
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i-1] and ami[i] < ami[i+1]:
            return i + 1, ami  # +1 because lag starts at 1

    return np.argmin(ami) + 1, ami  # fallback if no local min

def compute_le(x,ms=None):
    mf = mf_value(x)
    if ms is None:
        min_tsep = int(round(1/mf))
    else:
        min_tsep = ms
    lag,_ = compute_ami_lag(x,bins = 100)
    #print(min_tsep,lag)
    le = nolds.lyap_r(x,min_tsep=min_tsep,lag=lag)
    return le

# --- Load from Hugging Face ---

url = "https://huggingface.co/datasets/zyllab/TTMs_on_MG/resolve/main/rolling_prediction.pkl"
data = load_data_from_hf(url)
df = pd.DataFrame(data)

les_df = load_les_data()


# --- Sidebar Inputs ---
st.sidebar.title("If TTM learn chaos?")
st.sidebar.write("### Parameters")
idx = st.sidebar.slider("Index", 0, 905, 40)
tau = st.sidebar.selectbox("Tau", [60, 120, 200])
frac = st.sidebar.selectbox("frac", [5, 30, 75])
location = st.sidebar.selectbox("location", ["last", "uniform"])
delay = st.sidebar.selectbox("Delay", [1, 2, 3])

# --- Filter by parameters ---
df_filtered = df[
    (df['tau'] == tau) &
    (df['frac'] == frac) &
    (df['location'] == location)
]

les_df_filtered = les_df[
    (les_df['tau'] == tau)
]

# --- Extract output vectors ---
les_org_diff_x0 = np.array(les_df_filtered['dif_x0'].tolist())[0,0]
les_org_diff_idx = np.array(les_df_filtered['dif_idx'].tolist())[0,0]
true_preds = np.array(df_filtered['True'].tolist())[0,0]
preds = np.array(df_filtered['Pred'].tolist())[0,0]
preds_pt = np.array(df_filtered['Pred_pt'].tolist())[0,0]
# --- MSE ---
mse_original = np.mean((preds - true_preds) ** 2)
mse_perturbed = np.mean((preds_pt - true_preds) ** 2)

st.write(f"### MSE Comparison")
st.write(f"- Original MSE: {mse_original:.4f}")
st.write(f"- Perturbed MSE: {mse_perturbed:.4f}")

# --- Lyapunov Exponents ---
np.random.seed(0)
les_pred_1 = []
les_pred_2 = []
les_pred_pt = []

for i in range(906):
    if (i%15==0):
        if (tau == 200):
            les_pred_1.append(compute_le(preds[i][576:1152],ms=200))
            les_pred_2.append(compute_le(preds[i][1056:1632],ms=200))
            les_pred_pt.append(compute_le(preds_pt[i][1056:1632],ms=200))
        else:
            les_pred_1.append(compute_le(preds[i][576:576+500]))
            les_pred_2.append(compute_le(preds[i][1056:1056+500]))
            les_pred_pt.append(compute_le(preds_pt[i][1056:1056+500]))


# --- Lyapunov Scatter Plots ---
st.write("## Lyapunov Exponents Comparison")

fig, axs = plt.subplots(2, 3, figsize=(18, 8))


min_len = min(len(les_org_diff_x0), len(les_pred_1))
x_vals = np.arange(min_len)

labels = ["les_pred_1", "les_pred_2", "les_pred_pt"]
les_preds = [les_pred_1, les_pred_2, les_pred_pt]

# x0 base
for i in range(3):
    axs[0, i].scatter(x_vals, les_org_diff_x0[:min_len], label="Org (x0)", alpha=0.5)
    axs[0, i].scatter(x_vals, les_preds[i][:min_len], label=labels[i], alpha=0.5)
    axs[0, i].axhline(np.mean(les_org_diff_x0[:min_len]), color='blue', linestyle='--', label="Org Mean")
    axs[0, i].axhline(np.mean(les_preds[i][:min_len]), color='orange', linestyle='--', label=f"{labels[i]} Mean")
    axs[0, i].axhline(0, color='black', linestyle=':')
    axs[0, i].legend()
    axs[0, i].set_title(f"x0 base vs {labels[i]}")

# idx base
min_len_idx = min(len(les_org_diff_idx), len(les_pred_1))
x_vals_idx = np.arange(min_len_idx)
for i in range(3):
    axs[1, i].scatter(x_vals_idx, les_org_diff_idx[:min_len_idx], label="Org (idx)", alpha=0.5)
    axs[1, i].scatter(x_vals_idx, les_preds[i][:min_len_idx], label=labels[i], alpha=0.5)
    axs[1, i].axhline(np.mean(les_org_diff_idx[:min_len_idx]), color='blue', linestyle='--', label="Org Mean")
    axs[1, i].axhline(np.mean(les_preds[i][:min_len_idx]), color='orange', linestyle='--', label=f"{labels[i]} Mean")
    axs[1, i].axhline(0, color='black', linestyle=':')
    axs[1, i].legend()
    axs[1, i].set_title(f"idx base vs {labels[i]}")

st.pyplot(fig)

# --- Time Series Plot ---
st.write("## Time Series Comparison")

fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
timesteps = np.arange(len(true_preds[idx]))

ax_ts.plot(timesteps, true_preds[idx], label="True", linewidth=1.5)
ax_ts.plot(timesteps, preds[idx], label="Pred", linestyle="--")
ax_ts.plot(timesteps, preds_pt[idx], label="Pred Perturbed", linestyle=":")
ax_ts.axvline(1056, color='red', linestyle='--', label="Cut Point @1056")

ax_ts.set_title("Time Series Comparison")
ax_ts.set_xlabel("Timestep")
ax_ts.set_ylabel("Value")
ax_ts.legend()
st.pyplot(fig_ts)


import plotly.graph_objects as go

# --- 3D Delay Embeddings ---
st.write("## 3D Delay Embedding")

def plot_3d_embedding(series, delay, label, rgb_color, opacity=0.7):
    emb = delay_embedding_3d(series, delay)
    return go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode='lines',
        marker=dict(
            size=2,
            color=rgb_color,     
            opacity=opacity       
        ),
        name=label
    )


# Embedding data
true_emb = plot_3d_embedding(true_preds[idx], delay, "True", 'lightskyblue', opacity=0.7)
pred_emb = plot_3d_embedding(preds[idx], delay, "Pred", 'orange', opacity=0.7)
pt_emb = plot_3d_embedding(preds_pt[idx], delay, "Pred Perturbed", 'orange', opacity=0.7)

# Plot 1: True vs Pred
fig_3d_1 = go.Figure(data=[true_emb, pred_emb])
fig_3d_1.update_layout(title="3D Embedding: True vs Pred")

# Plot 2: True vs Perturbed Pred
fig_3d_2 = go.Figure(data=[true_emb, pt_emb])
fig_3d_2.update_layout(title="3D Embedding: True vs Perturbed Pred")

st.plotly_chart(fig_3d_1)
st.plotly_chart(fig_3d_2)

