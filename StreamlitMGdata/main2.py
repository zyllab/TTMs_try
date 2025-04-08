import streamlit as st
import numpy as np
import plotly.graph_objects as go
import nolds

# ========== Equation and Integration ==========
st.markdown(r"""
### Mackey-Glass Equation

$$
\frac{dx(t)}{dt} = \beta \cdot \frac{x(t - \tau)\,\theta^n}{\theta^n + x(t - \tau)^n} - \gamma \cdot x(t)
$$

---

### Numerical Methods

**Euler Method**:
            
$$
 x(t + \Delta t) = x(t) + \Delta t \cdot f(x(t), x(t - \tau))
$$

**RK4 Method (fixed delay)**:
            
$$
\begin{aligned}
    k_1 &= f(x_n, x_{n-\tau}) \\
    k_2 &= f\left(x_n + \frac{\Delta t}{2}k_1, x_{n-\tau}\right) \\
    k_3 &= f\left(x_n + \frac{\Delta t}{2}k_2, x_{n-\tau}\right) \\
    k_4 &= f\left(x_n + \Delta t\cdot k_3, x_{n-\tau}\right) \\
    x_{n+1} &= x_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

**RK4 Interpolation Method**:
            
$$
\begin{aligned}
    \tilde{x}_{n-\tau} &= \frac{x_{n-\tau} + x_{n-\tau+1}}{2} \\
    k_1 &= f(x_n, x_{n-\tau}) \\
    k_2 &= f\left(x_n + \frac{\Delta t}{2}k_1, \tilde{x}_{n-\tau}\right) \\
    k_3 &= f\left(x_n + \frac{\Delta t}{2}k_2, \tilde{x}_{n-\tau}\right) \\
    k_4 &= f\left(x_n + \Delta t\cdot k_3, x_{n-\tau+1}\right) \\
    x_{n+1} &= x_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$
""")

# ========== Equation and Integration ==========
def MG_eq(x, x_pre, gamma, beta, theta, n):
    return x_pre * beta * (theta**n) / (theta**n + x_pre**n) - gamma * x

def MG_rk4(x, x_pre, gamma, beta, theta, n, delta):
    k1 = MG_eq(x, x_pre, gamma, beta, theta, n)
    k2 = MG_eq(x + delta * k1 / 2, x_pre, gamma, beta, theta, n)
    k3 = MG_eq(x + delta * k2 / 2, x_pre, gamma, beta, theta, n)
    k4 = MG_eq(x + delta * k3, x_pre, gamma, beta, theta, n)
    return x + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def MG_rk4_interp(x, x_pre, x_pre_f, delta, gamma, beta, theta, n):
    inter = (x_pre + x_pre_f) / 2
    k1 = MG_eq(x, x_pre, gamma, beta, theta, n)
    k2 = MG_eq(x + delta * k1 / 2, inter, gamma, beta, theta, n)
    k3 = MG_eq(x + delta * k2 / 2, inter, gamma, beta, theta, n)
    k4 = MG_eq(x + delta * k3, x_pre_f, gamma, beta, theta, n)
    return x + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def MG_euler(x, x_pre, delta, gamma, beta, theta, n):
    return x + delta * MG_eq(x, x_pre, gamma, beta, theta, n)

# ========== Data Generator ==========
def generate_MG(method, gamma, beta, tau, theta, n, x0, N, delta, history_value):
    past_len = int(np.floor(tau / delta))
    x_past = np.zeros(past_len+N+1)+history_value
    x=x0
    T = np.zeros(N + 1)
    X = np.zeros(N + 1)
    time = 0

    for i in range(N + 1):
        X[i] = x
        T[i] = time
        x_pre = x_past[i]
        x_pre_f = x_past[i + 1]

        if method == "RK4":
            x_delta = MG_rk4(x, x_pre, gamma, beta, theta, n, delta)
        elif method == "RK4 Interp":
            x_delta = MG_rk4_interp(x, x_pre, x_pre_f, delta, gamma, beta, theta, n)
        elif method == "Euler":
            x_delta = MG_euler(x, x_pre, delta, gamma, beta, theta, n)

        x_past[past_len + i] = x_delta
        x = x_delta
        time += delta

    return T, X

# ========== Streamlit App ==========
st.title("Mackey-Glass Simulator with Multiple Methods")

st.sidebar.header("Parameters")
method = st.sidebar.selectbox("Integration Method", ["RK4", "RK4 Interp", "Euler"])
gamma = st.sidebar.slider("gamma", 0.0, 1.0, 0.1)
beta = st.sidebar.slider("beta", 0.0, 10.0, 0.2)
tau = st.sidebar.slider("tau", 0, 400, 23, step=1)
theta = st.sidebar.slider("theta", 0.1, 5.0, 1.0)
n = st.sidebar.slider("n", 1, 20, 10)
x0 = st.sidebar.slider("x0 (initial value)", 0.0, 16.0, 0.2)
N = st.sidebar.number_input("N (simulation length)", min_value=1000, max_value=1000000, value=1000000)
delta = st.sidebar.slider("delta (step size)", 0.001, 1.0, 0.01)
history_value = st.sidebar.number_input("Initial value in [-tau, 0]", min_value=0.0, max_value=20.0, value=0.1)

T, X = generate_MG(method, gamma, beta, tau, theta, n, x0, int(N), delta, history_value)

st.subheader("Generated Time Series")
fig = go.Figure()
fig.add_trace(go.Scatter(x=T[::100], y=X[::100], mode='lines', name='x(t)'))
fig.update_layout(title="Mackey-Glass Time Series", xaxis_title="t", yaxis_title="x(t)")
st.plotly_chart(fig, use_container_width=True)

# ========== Lyapunov Exponent ==========
st.subheader("Estimated Lyapunov Exponent")
sample = X[::100][9000:10000]  # Downsample and use stable part

try:
    fft_vals = np.fft.rfft(sample)
    psd = np.abs(fft_vals)**2
    freqs = np.fft.rfftfreq(len(sample))
    mean_freq = np.sum(freqs[1:] * psd[1:]) / np.sum(psd[1:])
    mean_period = 1 / mean_freq
    min_tsep = int(round(mean_period))

    lyap_est = nolds.lyap_r(sample, emb_dim=10, min_tsep=min_tsep)
    st.write(f"Lyapunov Exponent (estimated): {lyap_est:.5f}")
except Exception as e:
    st.error(f"Failed to estimate Lyapunov exponent: {e}")

# ========== 2D Delay Embedding ==========
st.subheader("2D Delay Embedding")
delay_embed_tau = st.sidebar.slider("Delay Embedding Step (points)", 1, 500, 100)
max_index_2d = len(X) - delay_embed_tau
if max_index_2d <= 0:
    st.warning("Not enough data points for the selected delay embedding step.")
else:
    X0_2d = X[:max_index_2d]
    X1_2d = X[delay_embed_tau : max_index_2d + delay_embed_tau]
    downsample_step_2d = st.sidebar.slider("Downsample for 2D plot", 1, 200, 1)
    X0_2d_plot = X0_2d[::downsample_step_2d]
    X1_2d_plot = X1_2d[::downsample_step_2d]

    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(
        x=X0_2d_plot,
        y=X1_2d_plot,
        mode='lines', 
        name='2D Embedding'
    ))
    fig_2d.update_layout(
        title="2D Delay Embedding",
        xaxis_title="X(t)",
        yaxis_title=f"X(t + {delay_embed_tau}\u0394t)"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# ========== 3D Delay Embedding ==========
st.subheader("3D Delay Embedding")



embed_dim = 3


max_index = len(X) - (embed_dim - 1)*delay_embed_tau
if max_index <= 0:
    st.warning("Not enough data points for the selected delay embedding step.")
else:
    X0 = X[:max_index]
    X1 = X[delay_embed_tau: max_index + delay_embed_tau]
    X2 = X[2*delay_embed_tau: max_index + 2*delay_embed_tau]
    
    
    downsample_step = st.sidebar.slider("Downsample for 3D plot", 1, 200, 10)
    X0_plot = X0[::downsample_step]
    X1_plot = X1[::downsample_step]
    X2_plot = X2[::downsample_step]

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=X0_plot,
        y=X1_plot,
        z=X2_plot,
        mode='markers',
        marker=dict(size=2),
        name="Delay Embedding"
    ))
    fig_3d.update_layout(
        scene = dict(
            xaxis_title="X(t)",
            yaxis_title=f"X(t + {delay_embed_tau}\u0394t)",
            zaxis_title=f"X(t + 2*{delay_embed_tau}\u0394t)"
        ),
        title="3D Delay Embedding"
    )
    st.plotly_chart(fig_3d, use_container_width=True)