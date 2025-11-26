# ============================================================
# IMPORTS & GLOBAL SETTINGS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# STEP 1: DATA LOADING & PREPARATION
# ============================================================

def load_umist_mat(mat_path):
    """
    Load the UMIST cropped face dataset.

    umist_cropped.mat structure (from inspection):
        - 'facedat': shape (1, 20), each cell is an array (112, 92, n_i)
                     for person i
        - 'dirnames': shape (1, 20), names for each person (not strictly needed)

    Returns:
        X: (N, 112*92) float32, flattened grayscale images
        y: (N,) int labels 0..19 (person index)
    """
    mat = loadmat(mat_path)
    facedat = mat['facedat']         
    dirnames = mat['dirnames'][0]    

    images = []
    labels = []

    num_people = facedat.shape[1]  
    for person_idx in range(num_people):
       
        person_imgs = facedat[0, person_idx]
        H, W, num_imgs = person_imgs.shape

        for j in range(num_imgs):
            img = person_imgs[:, :, j]  
            images.append(img.flatten().astype(np.float32))
            labels.append(person_idx)    

    X = np.vstack(images)                # (N, 112*92)
    y = np.array(labels, dtype=int)      # (N,)

    print("Loaded UMIST faces:")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)
    print("  #persons:", num_people)

    return X, y


def create_dataframe(X, y):
    """
    Create a Pandas DataFrame with 'image' (flattened vector) and 'label'.
    """
    df = pd.DataFrame({
        'image': list(X),
        'label': y
    })
    print("\nDataFrame head:")
    print(df.head())
    return df


def show_sample_images(X, y, n_samples=9, img_shape=(112, 92)):
    """
    Show a grid of sample images with their labels.
    """
    n_samples = min(n_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
    axes = axes.ravel()

    for ax, idx in zip(axes, indices):
        img = X[idx].reshape(img_shape)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {y[idx]}")
        ax.axis('off')

  
    for ax in axes[n_samples:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 2: STRATIFIED SPLIT + NORMALIZATION + DISTRIBUTION PLOTS
# ============================================================

def stratified_split_normalize(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Stratified split into train/val/test and apply StandardScaler.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled
        y_train, y_val, y_test
        scaler
        X_train_orig, X_val_orig, X_test_orig  (unscaled, for visualization)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

  
    X_train_orig, X_tmp_orig, y_train, y_tmp = train_test_split(
        X, y,
        test_size=1.0 - train_ratio,
        stratify=y,
        random_state=RANDOM_STATE
    )

  
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val_orig, X_test_orig, y_val, y_test = train_test_split(
        X_tmp_orig, y_tmp,
        test_size=1.0 - val_size,
        stratify=y_tmp,
        random_state=RANDOM_STATE
    )

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_val_scaled = scaler.transform(X_val_orig)
    X_test_scaled = scaler.transform(X_test_orig)

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            scaler,
            X_train_orig, X_val_orig, X_test_orig)


def plot_label_distribution(y_train, y_val, y_test, title_prefix=""):
    """
    Plot the distribution of images per person (label) for each subset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, y, subset_name in zip(
            axes,
            [y_train, y_val, y_test],
            ["Train", "Validation", "Test"]):
        counts = pd.Series(y).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"{title_prefix}{subset_name} Distribution")
        ax.set_xlabel("Person (label)")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 3: DIMENSIONALITY REDUCTION (PCA + AUTOENCODER)
# ============================================================

def run_pca(X_train, X_val, X_test, variance_threshold=0.95):
  
    # Fit PCA with all components to see the variance curve
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_components_needed = np.searchsorted(cumulative, variance_threshold) + 1

    print(f"PCA: variance threshold = {variance_threshold}")
    print(f"Number of components needed = {n_components_needed}")
    print(f"Actual cumulative variance = {cumulative[n_components_needed-1]:.4f}")

    # Plot cumulative explained variance
    max_to_show = min(150, len(cumulative))
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, max_to_show + 1), cumulative[:max_to_show], marker='o')
    plt.axhline(variance_threshold, linestyle='--', label=f"threshold={variance_threshold}")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA – Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Fit PCA again with the chosen number of components
    pca_final = PCA(n_components=n_components_needed, random_state=RANDOM_STATE)
    Z_train = pca_final.fit_transform(X_train)
    Z_val = pca_final.transform(X_val)
    Z_test = pca_final.transform(X_test)

    # 2D projection just for visualization
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_train_2d = pca_2d.fit_transform(X_train)

    return pca_final, (Z_train, Z_val, Z_test), (pca_2d, Z_train_2d)


    return pca_final, (Z_train, Z_val, Z_test), (pca_2d, Z_train_2d)


def plot_2d_embedding(Z_2d, labels, title="2D Projection", cmap='tab20'):
    """
    Generic 2D scatter plot for an embedding (e.g. PCA, autoencoder).
    """
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                          c=labels, cmap=cmap, s=10, alpha=0.8)
    plt.colorbar(scatter, label='Label')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()


def build_autoencoder(input_dim, latent_dim=48):  
    """
    Simple fully connected autoencoder.
    """
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.Dense(128, activation='relu')(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)

    x = layers.Dense(128, activation='relu')(latent)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = models.Model(input_layer, output_layer, name="autoencoder")
    encoder = models.Model(input_layer, latent, name="encoder")

    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )

    return autoencoder, encoder



def train_autoencoder(X_train, X_val, latent_dim=48, epochs=50, batch_size=64):
    """
    Train the autoencoder and return (autoencoder, encoder).
    Also plots the reconstruction loss curves.
    """
    input_dim = X_train.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)

    es = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    # Plot reconstruction loss curves
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return autoencoder, encoder

def show_autoencoder_reconstructions(autoencoder, X_data, img_shape=(112, 92), n_samples=6):
    """
    Show original vs reconstructed images to visually inspect AE quality.
    """
    n_samples = min(n_samples, len(X_data))
    idxs = np.random.choice(len(X_data), n_samples, replace=False)

    X_orig = X_data[idxs]
    X_recon = autoencoder.predict(X_orig)

    n_rows = 2
    n_cols = n_samples

    plt.figure(figsize=(2 * n_cols, 4))

    # Original images
    for i, idx in enumerate(idxs):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(X_orig[i].reshape(img_shape), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

    # Reconstructed images
    for i, idx in enumerate(idxs):
        ax = plt.subplot(n_rows, n_cols, n_cols + i + 1)
        ax.imshow(X_recon[i].reshape(img_shape), cmap='gray')
        ax.set_title("Recon")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN: RUN STEPS 1–3 FOR DEMO
# ============================================================

def main():
    mat_path = "umist_cropped.mat"   

    # ----- STEP 1 -----
    X, y = load_umist_mat(mat_path)
    df = create_dataframe(X, y)
    show_sample_images(X, y, n_samples=9, img_shape=(112, 92))

    # ----- STEP 2 -----
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler,
     X_train_orig, X_val_orig, X_test_orig) = stratified_split_normalize(X, y)

    plot_label_distribution(y_train, y_val, y_test,
                            title_prefix="UMIST - ")

    # ----- STEP 3A: PCA -----
    pca, (Z_train_pca, Z_val_pca, Z_test_pca), (pca_2d, Z_train_2d) = \
    run_pca(X_train, X_val, X_test, variance_threshold=0.95)

    plot_2d_embedding(Z_train_2d, y_train,
                      title="PCA 2D Projection (Train Labels)")

    # ----- STEP 3B: AUTOENCODER -----
    autoencoder, encoder = train_autoencoder(
    X_train, X_val, latent_dim=48, epochs=50, batch_size=64
)
    show_autoencoder_reconstructions(autoencoder, X_val, img_shape=(112, 92))


    Z_train_ae = encoder.predict(X_train)
    Z_train_ae_2d = Z_train_ae[:, :2]
    plot_2d_embedding(Z_train_ae_2d, y_train,
                      title="Autoencoder Latent 2D (Train Labels)")


if __name__ == "__main__":
    main()
