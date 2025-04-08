import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



def MG_eq(x, x_pre, gamma, beta, theta, n):
    return x_pre * beta * (theta**n) / (theta**n + x_pre**n) - gamma * x

def MG_rk4(x, x_pre, gamma, beta, theta, n, delta):
    k1 = MG_eq(x, x_pre, gamma, beta, theta, n)
    k2 = MG_eq(x + delta * k1 / 2, x_pre, gamma, beta, theta, n)
    k3 = MG_eq(x + delta * k2 / 2, x_pre, gamma, beta, theta, n)
    k4 = MG_eq(x + delta * k3, x_pre, gamma, beta, theta, n)
    return x + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def MG_generate(gamma, beta, tau, theta, n, x0, N, delta):
    past_len = int(np.floor(tau / delta))
    x_past = np.zeros(past_len)
    X = np.zeros(N + 1)
    T = np.zeros(N + 1)
    x = x0
    time = 0
    index = 0

    for i in range(N + 1):
        X[i] = x

        if tau == 0:
            x_pre = 0
        else:
            x_pre = x_past[index]

        x_delta = MG_rk4(x, x_pre, gamma, beta, theta, n, delta)

        if tau != 0:
            x_past[index] = x_delta
            index = (index + 1) % past_len

        T[i] = time
        time += delta
        x = x_delta

    return T, X


st.title("Mackey-Glass Time Series Simulator")

gamma = st.sidebar.slider("gamma", 0.0, 1.0, 0.2)
beta = st.sidebar.slider("beta", 0.0, 1.0, 0.2)
tau = st.sidebar.slider("tau", 0, 200, 30,step=1)
theta = st.sidebar.slider("theta", 0.1, 5.0, 0.2)
n = st.sidebar.slider("n", 1, 20, 10)
x0 = st.sidebar.slider("x0", 0.0, 1.0, 0.2)
N = st.sidebar.number_input("N (simulation length)", min_value=1000, max_value=1000000, value=100000)
delta = st.sidebar.slider("delta (step size)", 0.001, 1.0, 0.01)


T, X = MG_generate(gamma, beta, tau, theta, n, x0, int(N), delta)


plot_range = 10000
step = 100
st.subheader("Mackey-Glass Time Series Plot")
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=T[::step][:plot_range], y=X[::step][:plot_range], mode='lines', name='x(t)'))
fig.update_layout(
    title="Mackey-Glass Time Series",
    xaxis_title="t",
    yaxis_title="x(t)",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)
