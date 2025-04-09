import streamlit as st
import pandas as pd
import numpy as np
import nolds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import StringIO
import requests

# --- Utils ---
def clean_and_parse(s):
    s = s.replace('\n', ' ')      # Replace newlines with spaces
    s = s.strip('[] ')            # Remove square brackets and surrounding spaces
    return np.fromstring(s, sep=' ')

def delay_embedding_3d(series, delay=1):
    return np.column_stack([
        series[:-2 * delay],
        series[delay:-delay],
        series[2 * delay:]
    ])

# --- Sidebar Inputs ---
st.sidebar.title("Time Series Forecasting Explorer")
st.sidebar.write("### Parameters")
idx = st.sidebar.slider("Index", 0, 905, 40)
tau = st.sidebar.selectbox("Tau", [60, 120, 200])
x0 = st.sidebar.selectbox("x0", [0.2, 1, 5, 8])
frac = st.sidebar.selectbox("frac", [5, 30, 75])
location = st.sidebar.selectbox("location", ["last", "uniform"])
delay = st.sidebar.selectbox("Delay", [1, 2, 3])


# --- Load from Hugging Face ---
url = "https://huggingface.co/datasets/zyllab/TTMs_on_MG/resolve/main/merged_full_dset.csv"
response = requests.get(url)

# Read CSV from the downloaded content
csv_data = StringIO(response.text)
df = pd.read_csv(csv_data)

df['past_target'] = df['past_target'].apply(clean_and_parse)
df['future_target'] = df['future_target'].apply(clean_and_parse)
df['pred'] = df['pred'].apply(clean_and_parse)
df['pred_per'] = df['pred_per'].apply(clean_and_parse)

# --- Filter by parameters ---
df_filtered = df[
    (df['x0'] == x0) &
    (df['tau'] == tau) &
    (df['frac'] == frac) &
    (df['location'] == location)
]

# --- Extract output vectors ---
past_targets = np.stack(df_filtered['past_target'].values)
future_targets = np.stack(df_filtered['future_target'].values)
pred = np.stack(df_filtered['pred'].values)
pred_per = np.stack(df_filtered['pred_per'].values)
st.write(f"### Data Shape: {past_targets.shape}, {future_targets.shape}, {pred.shape}, {pred_per.shape}")
# --- MSE ---
mse_original = np.mean((pred - future_targets) ** 2)
mse_perturbed = np.mean((pred_per - future_targets) ** 2)

st.write(f"### MSE Comparison")
st.write(f"- Original MSE: {mse_original:.4f}")
st.write(f"- Perturbed MSE: {mse_perturbed:.4f}")

# --- Lyapunov Exponents ---
pred_lyap = nolds.lyap_r(pred[idx], emb_dim=10, min_tsep=33)
pred_per_lyap = nolds.lyap_r(pred_per[idx], emb_dim=10, min_tsep=33)
true_lyap = nolds.lyap_r(future_targets[idx], emb_dim=10, min_tsep=33)
past_lyap = nolds.lyap_r(past_targets[idx][-96:], emb_dim=10, min_tsep=33)
data = np.concatenate([past_targets[idx], future_targets[idx]])
fft_vals = np.fft.rfft(data)
psd = np.abs(fft_vals)**2
freqs = np.fft.rfftfreq(len(data))
# Skip DC component at index 0
mean_freq = np.sum(freqs[1:] * psd[1:]) / np.sum(psd[1:])
mean_period = 1 / mean_freq
min_tsep = int(round(mean_period))
overall_lyap = nolds.lyap_r(data, emb_dim=10, min_tsep=min_tsep)

st.write(f"### Lyapunov Exponents")
st.write(f"- Overall Lyapunov Exponent: `{overall_lyap:.4f}`")
st.write(f"- True Prediction Lyapunov Exponent: `{true_lyap:.4f}`")
st.write(f"- Last 96 of Histroy Lyapunov Exponent: `{past_lyap:.4f}`")
st.write(f"- Prediction Lyapunov Exponent: `{pred_lyap:.4f}`")
st.write(f"- Perturbed Prediction Lyapunov Exponent: `{pred_per_lyap:.4f}`")


# --- 2D Plot ---
t_full = np.arange(608)
t_pred = np.arange(512, 608)

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(t_full[:512], past_targets[idx], label='Observed', color='blue')
ax1.plot(t_pred, future_targets[idx], label='True Value', color='green')
ax1.plot(t_pred, pred[idx], label='Prediction', color='red', linestyle='--')
ax1.plot(t_pred, pred_per[idx], label='Prediction Perturbed', color='gray', linestyle='--',alpha = 0.5)

ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_title('Time Series Forecasting')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- 3D Delay Embedding Plot ---

import plotly.graph_objects as go
gt_segment = np.concatenate([past_targets[idx], future_targets[idx]])
pred_segment = pred[idx]

gt_embed = delay_embedding_3d(gt_segment, delay)
pred_embed = delay_embedding_3d(pred_segment, delay)
pred_per_embed = delay_embedding_3d(pred_per[idx], delay)

# fig2 = plt.figure(figsize=(12, 7))
# ax2 = fig2.add_subplot(111, projection='3d')

# ax2.plot(gt_embed[:, 0], gt_embed[:, 1], gt_embed[:, 2], label='Ground Truth', alpha=0.4)
# #ax2.scatter(gt_embed[:, 0], gt_embed[:, 1], gt_embed[:, 2], label='Ground Truth', alpha=0.4, s=5)
# ax2.plot(pred_embed[:, 0], pred_embed[:, 1], pred_embed[:, 2], label='Prediction', alpha=0.7, color='red')
# ax2.plot(pred_per_embed[:, 0], pred_per_embed[:, 1], pred_per_embed[:, 2], label='Perturbed Prediction', alpha=0.2)

# ax2.set_xlabel('x(t)')
# ax2.set_ylabel('x(t+delay)')
# ax2.set_zlabel('x(t+2delay)')
# ax2.set_title('3D Delay Embedding')
# ax2.legend()

# st.pyplot(fig2)


fig = go.Figure()

# Ground Truth
fig.add_trace(go.Scatter3d(
    x=gt_embed[:, 0],
    y=gt_embed[:, 1],
    z=gt_embed[:, 2],
    mode='lines',
    name='Ground Truth',
    line=dict(color='blue', width=3),
    opacity=0.4
))

# Prediction
fig.add_trace(go.Scatter3d(
    x=pred_embed[:, 0],
    y=pred_embed[:, 1],
    z=pred_embed[:, 2],
    mode='lines',
    name='Prediction',
    line=dict(color='red', width=4),
    opacity=0.7
))

# Perturbed Prediction
fig.add_trace(go.Scatter3d(
    x=pred_per_embed[:, 0],
    y=pred_per_embed[:, 1],
    z=pred_per_embed[:, 2],
    mode='lines',
    name='Perturbed Prediction',
    line=dict(color='green', width=2),
    opacity=0.2
))

# Layout settings
fig.update_layout(
    title='3D Delay Embedding',
    scene=dict(
        xaxis_title='x(t)',
        yaxis_title='x(t + delay)',
        zaxis_title='x(t + 2*delay)'
    ),
    width=900,
    height=600,
    legend=dict(x=0, y=1)
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
