import streamlit as st
import pickle
import matplotlib.pyplot as plt
import requests

# Function to download the pickle file from Hugging Face
def download_pickle_file(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        return pickle.loads(response.content)
    else:
        st.error("Failed to download the file.")
        return None

# Streamlit app
st.title("Time Series Prediction Plots")

# User input to select the index to plot
index_to_plot = st.number_input("Enter the index to plot", min_value=0, value=10, step=1)

# URL for the combined_data.pkl file uploaded to Hugging Face
file_url = 'https://huggingface.co/datasets/zyllab/TTMs_on_MG/resolve/main/combined_data.pkl'  # Replace with your actual file URL

# Download and load the combined data
combined_data = download_pickle_file(file_url)

if combined_data:
    # Define the true sequences for comparison
    true_60 = combined_data['true_60']
    true_200 = combined_data['true_200']

    # Define the length of the context (the part before the prediction)
    context_length = 512  # Context length before prediction starts

    # List of predictions for the first 3 plots (against true_200)
    pred_200_keys = ['TTM_pred_200','TTM_5_pred_200','TTM_short_pred_200']

    # List of predictions for the last 4 plots (against true_60)
    pred_60_keys = ['LSTM_pred','TTM_pred','TTM_5_pred','TTM_short_pred']

    # Create a figure with subplots for 7 graphs, changing layout to 7 rows and 1 column
    fig, axes = plt.subplots(7, 1, figsize=(10, 25))

    # Plot the first 3 predictions against true_200
    for i, key in enumerate(pred_200_keys):
        axes[i].plot(range(len(true_200[index_to_plot])), true_200[index_to_plot], label='True_200', color='gray',alpha = 0.5)
        
        # Add dashed line to separate context and prediction
        axes[i].axvline(x=context_length, linestyle='--', color='black', label='Prediction Start')
        
        # Plot the predictions from the context onward
        axes[i].plot(range(context_length, context_length + len(combined_data[key][index_to_plot])), combined_data[key][index_to_plot], label=key)
        axes[i].set_title(f"True vs Prediction ({key}) - True_200")
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Lyapunov Exponent')
        axes[i].legend()

    # Plot the last 4 predictions against true_60
    for i, key in enumerate(pred_60_keys):
        axes[i + 3].plot(range(len(true_60[index_to_plot])), true_60[index_to_plot], label='True_60', color='gray',alpha = 0.5)
        
        # Add dashed line to separate context and prediction
        axes[i + 3].axvline(x=context_length, linestyle='--', color='black', label='Prediction Start')
        
        # Plot the predictions from the context onward
        axes[i + 3].plot(range(context_length, context_length + len(combined_data[key][index_to_plot])), combined_data[key][index_to_plot], label=key)
        axes[i + 3].set_title(f"True vs Prediction ({key}) - True_60")
        axes[i + 3].set_xlabel('Time Step')
        axes[i + 3].set_ylabel('Lyapunov Exponent')
        axes[i + 3].legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Please make sure the data file is accessible.")
