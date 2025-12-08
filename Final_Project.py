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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
    classification_report
)

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
    dirnames = mat['dirnames'][0]  # not used, but kept if needed

    images = []
    labels = []

    num_people = facedat.shape[1]
    for person_idx in range(num_people):

        person_imgs = facedat[0, person_idx]  # shape (112, 92, n_i)
        H, W, num_imgs = person_imgs.shape

        for j in range(num_imgs):
            img = person_imgs[:, :, j]
            images.append(img.flatten().astype(np.float32))
            labels.append(person_idx)

    X = np.vstack(images)           # (N, 112*92)
    y = np.array(labels, dtype=int) # (N,)

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

    # Turn off any unused axes
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

    # First split off the training set
    X_train_orig, X_tmp_orig, y_train, y_tmp = train_test_split(
        X, y,
        test_size=1.0 - train_ratio,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Then split tmp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val_orig, X_test_orig, y_val, y_test = train_test_split(
        X_tmp_orig, y_tmp,
        test_size=1.0 - val_size,
        stratify=y_tmp,
        random_state=RANDOM_STATE
    )

    # Normalization (fit on train only)
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
    """
    Run PCA and choose number of components to reach given variance threshold.
    Also returns a 2D PCA projection for plotting.
    """

    # Fit PCA with all components to see the variance curve
    pca_full = PCA()
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
    pca_final = PCA(n_components=n_components_needed)
    Z_train = pca_final.fit_transform(X_train)
    Z_val = pca_final.transform(X_val)
    Z_test = pca_final.transform(X_test)

    # 2D projection just for visualization
    pca_2d = PCA(n_components=2)
    Z_train_2d = pca_2d.fit_transform(X_train)

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
    for i in range(n_samples):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(X_orig[i].reshape(img_shape), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

    # Reconstructed images
    for i in range(n_samples):
        ax = plt.subplot(n_rows, n_cols, n_cols + i + 1)
        ax.imshow(X_recon[i].reshape(img_shape), cmap='gray')
        ax.set_title("Recon")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 4: CLUSTERING HELPERS
# ============================================================

def compute_purity(y_true, y_pred):
    """
    Cluster purity: sum_j max_i |C_j ∩ L_i| / N
    where C_j are clusters and L_i are true classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape[0] == y_pred.shape[0]

    N = len(y_true)
    purity_sum = 0.0
    for cluster_id in np.unique(y_pred):
        idx = np.where(y_pred == cluster_id)[0]
        if len(idx) == 0:
            continue
        true_labels_in_cluster = y_true[idx]
        majority_count = np.bincount(true_labels_in_cluster).max()
        purity_sum += majority_count

    return purity_sum / N


def plot_clustering_2d(Z_2d, cluster_labels, title="2D Clustering"):
    """
    Scatter plot of a 2D embedding, colored by cluster labels.
    """
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                          c=cluster_labels, cmap='tab20', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()


def run_kmeans_clustering(Z, y, n_clusters_list, name="PCA"):
    """
    Run KMeans for multiple k and print metrics.
    Z: feature matrix (e.g., PCA-reduced)
    y: true labels
    """
    print(f"\n=== {name}: KMeans Clustering ===")
    last_labels = None
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(Z)
        last_labels = cluster_labels

        purity = compute_purity(y, cluster_labels)
        ari = adjusted_rand_score(y, cluster_labels)
        nmi = normalized_mutual_info_score(y, cluster_labels)
        sil = silhouette_score(Z, cluster_labels)

        print(f"k = {k:2d} | purity = {purity:.3f}, ARI = {ari:.3f}, "
              f"NMI = {nmi:.3f}, silhouette = {sil:.3f}")

    # Return labels for the last k (usually the “chosen” one)
    return last_labels


def run_agg_clustering(Z, y, n_clusters, linkage="ward", name="PCA"):
    """
    Run Agglomerative (hierarchical) clustering once, print metrics, and
    return cluster labels.
    """
    print(f"\n=== {name}: Agglomerative Clustering (linkage={linkage}) ===")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_labels = agg.fit_predict(Z)

    purity = compute_purity(y, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    sil = silhouette_score(Z, cluster_labels)

    print(f"n_clusters = {n_clusters:2d} | purity = {purity:.3f}, "
          f"ARI = {ari:.3f}, NMI = {nmi:.3f}, silhouette = {sil:.3f}")

    return cluster_labels


# ============================================================
# STEP 5: SUPERVISED LEARNING – NN CLASSIFIER
# ============================================================

def build_mlp_classifier(input_dim, num_classes):
    """
    Simple feedforward neural network classifier on top of
    reduced-dimension features (e.g., PCA).
    """
    model = models.Sequential(name="MLP_classifier")
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train_and_evaluate_classifier(Z_train, y_train,
                                  Z_val, y_val,
                                  Z_test, y_test):
    """
    Train the MLP classifier and evaluate on test set.
    Also plots train/val accuracy & loss, and prints confusion matrix.
    """
    num_classes = len(np.unique(y_train))
    input_dim = Z_train.shape[1]

    model = build_mlp_classifier(input_dim, num_classes)

    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        Z_train, y_train,
        validation_data=(Z_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    # Plot training curves
    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("NN Classifier – Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NN Classifier – Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(Z_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

    # Confusion matrix & classification report
    y_pred_probs = model.predict(Z_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    return model


# ============================================================
# MAIN: RUN STEPS 1–5
# ============================================================

def main():
    mat_path = "umist_cropped.mat"   # make sure this file is in your working dir

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

    # =======================================================
    # STEP 4: CLUSTERING ON REDUCED FEATURES
    # =======================================================
    num_classes = len(np.unique(y_train))

    # --- 4A: KMeans on PCA features ---
    k_list = [num_classes, num_classes + 3]  # e.g. 20 and 23
    kmeans_labels_pca = run_kmeans_clustering(
        Z_train_pca, y_train, n_clusters_list=k_list, name="PCA"
    )

    # Visualize clustering result (using PCA 2D embedding colored by cluster)
    plot_clustering_2d(Z_train_2d, kmeans_labels_pca,
                       title="KMeans Clusters on PCA (Train)")

    # --- 4B: Agglomerative clustering on PCA features ---
    agg_labels_pca = run_agg_clustering(
        Z_train_pca, y_train, n_clusters=num_classes,
        linkage="ward", name="PCA"
    )

    plot_clustering_2d(Z_train_2d, agg_labels_pca,
                       title="Agglomerative Clusters on PCA (Train)")

    # (Optional) Repeat clustering on autoencoder latent space if you like:
    # kmeans_labels_ae = run_kmeans_clustering(
    #     Z_train_ae, y_train, n_clusters_list=k_list, name="Autoencoder"
    # )
    # plot_clustering_2d(Z_train_ae_2d, kmeans_labels_ae,
    #                    title="KMeans Clusters on AE Latent (Train)")

    # =======================================================
    # STEP 5: SUPERVISED NN CLASSIFIER ON PCA FEATURES
    # =======================================================
    print("\n=== Training NN classifier on PCA features ===")
    classifier_model = train_and_evaluate_classifier(
        Z_train_pca, y_train,
        Z_val_pca, y_val,
        Z_test_pca, y_test
    )


if __name__ == "__main__":
    main()
