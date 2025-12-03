import streamlit as st
import pandas as pd
import numpy as np
from gcm import GCM
import os
import seaborn as sns

from matplotlib import pyplot as plt


def compute_multipole(df, columns, method='pearson'):
    corr_matrix = df[columns].corr(method=method)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    multipole = 1 - min(eigenvalues)
    return multipole
    # This stub just returns summary stats


def heatmap(df_train, df_synth):
    corr_real = np.corrcoef(df_train, rowvar=False)
    corr_synth = np.corrcoef(df_synth, rowvar=False)
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))

    sns.heatmap(corr_real, ax=axes[0], cmap="coolwarm", center=0, cbar=False)
    axes[0].set_title("Original Correlation Matrix")
    sns.heatmap(corr_synth, ax=axes[1], cmap="coolwarm", center=0, cbar=False)
    axes[1].set_title("Synthetic Correlation Matrix")
    return fig


def generate_synthetic_data(df_train, n_samples):
    gcm = GCM()
    gcm.fit(df_train)
    synth_data = pd.DataFrame(gcm.sample(num_samples=n_samples),
                              columns=df_train.columns)
    return synth_data


# ---------------------------------------------
# Streamlit App
# ---------------------------------------------
st.title("GCM Synthetic Data Generator")

# st.markdown("""
# This app allows you to:
# 1. Upload a CSV file **or** specify a directory containing CSV files
# 1. Load the data into a pandas DataFrame
# 2. Split into train/test
# 3. Generate synthetic data (your custom generator)
# 4. Validate synthetic data by comparing **multipoles**
# """)

# ==================================================
# 1. Upload CSV or choose directory
# ==================================================
st.header("Load Data")

upload_option = st.radio(
    "How do you want to load your data?",
    ["Upload CSV", "Use directory path"]
)

df = None

if upload_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded dataframe with shape {df.shape}")

else:
    dir_path = st.text_input("Enter directory path")
    if dir_path and os.path.isdir(dir_path):
        csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
        file_choice = st.selectbox("Select CSV file", csv_files)
        if file_choice:
            df = pd.read_csv(os.path.join(dir_path, file_choice))
            st.success(f"Loaded dataframe with shape {df.shape}")
    if dir_path and os.path.isfile(dir_path):
        df = pd.read_csv(dir_path)
        st.success(f"Loaded dataframe with shape {df.shape}")

if df is not None:
    st.dataframe(df.head())

# ==================================================
# 2. Train-test split
# ==================================================
if df is not None:
    st.header("Train/Test Split")

    st.session_state.train_pct = st.slider("Train set percentage:", 50, 95, 80)

    if st.button("Split Data"):
        train_df = df.sample(frac=st.session_state.train_pct / 100, random_state=42)
        test_df = df.drop(train_df.index)

        st.write(f"Train shape: {train_df.shape}")
        st.write(f"Test shape: {test_df.shape}")

        st.session_state["train_df"] = train_df
        st.session_state["test_df"] = test_df

# ==================================================
# 3. Generate Synthetic Data
# ==================================================
if "train_df" in st.session_state:
    st.header("Generate Synthetic Data")

    sample_size = st.number_input("Synthetic sample size:", 10, 1_000_000, 1000)

    if st.button("Generate Synthetic Dataset"):
        synth_df = generate_synthetic_data(st.session_state["train_df"], sample_size)
        st.session_state["synth_df"] = synth_df

        st.success("Synthetic data generated!")
        st.dataframe(synth_df.head())

        # Download button
        csv = synth_df.to_csv(index=False)
        st.download_button("Download Synthetic CSV", data=csv, file_name="synthetic_data.csv")

        # Optional save-to-path
        save_path = st.text_input("Save synthetic dataset to path (optional)")
        if save_path and st.button("Save File"):
            synth_df.to_csv(save_path, index=False)
            st.success(f"File saved to {save_path}")

        # Metadata export
        metadata = (
            "Synthetic Data Generation Metadata\n"
            f"Original Train Shape: {st.session_state['train_df'].shape}\n"
            f"Synthetic Shape: {synth_df.shape}\n"
            f"Sample Size: {sample_size}\n"
            f"Train/Test Split Percentage: {st.session_state.train_pct}%\n"
        )

        st.subheader("Metadata Export")

        # Download button
        st.download_button(
            label="Download Metadata",
            data=metadata,
            file_name="metadata.txt",
            mime="text/plain"
        )

        # Optional save-to-disk
        metadata_path = st.text_input("Save metadata to path (optional)")
        if metadata_path and st.button("Save Metadata to Disk"):
            with open(metadata_path, "w") as f:
                f.write(metadata)
            st.success(f"Metadata saved to {metadata_path}")

# ==================================================
# 4. Validate via multipole comparison
# ==================================================
if "synth_df" in st.session_state and "train_df" in st.session_state:
    st.header("Validate Synthetic Data")

    fig = heatmap(st.session_state["train_df"], st.session_state["synth_df"])
    st.pyplot(fig)
    cols = st.multiselect(
        "Select columns to compare:",
        st.session_state["train_df"].columns.tolist()
    )

    if len(cols) >= 2:
        if st.button("Compute Multipoles"):
            train_multipole = compute_multipole(st.session_state["train_df"], cols)
            synth_multipole = compute_multipole(st.session_state["synth_df"], cols)

            st.subheader("Train Multipoles")
            st.write(train_multipole)

            st.subheader("Synthetic Multipoles")
            st.write(synth_multipole)

            st.subheader("Difference (Synthetic - Train)")
            try:
                diff = synth_multipole - train_multipole
                st.write(diff)
            except:
                st.info("Cannot compute numeric differences for these multipole outputs.")
