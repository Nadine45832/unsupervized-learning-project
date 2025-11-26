import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow.keras import layers, models, callbacks, Sequential

RANDOM_STATE = 42

def load_data():
    mat = scipy.io.loadmat('umist_cropped.mat')
    facedat = mat['facedat']

    print(mat.keys())
    print("Face data shape", facedat.shape)
    print("Dirnames shape", facedat.shape)
    print("People identifiers", mat['dirnames'][0])

    num_people = facedat.shape[1]
    print("Amount of different faces", num_people)

    all_images = []
    all_labels = []
    images_array = facedat[0]

    for idx, subject in enumerate(images_array):
        images = subject

        # From (112,92,N) to (N, 112*92)
        N = images.shape[2]
        flat = images.reshape(-1, N).T
        all_images.append(flat.astype(np.float32))

        all_labels.extend([idx] * N)

    print("Amount of images", len(all_images))

    return np.vstack(all_images), all_labels


def show_sample_images(images, labels, images_to_show):
    indices = np.random.choice(images.shape[0], images_to_show, replace=False)

    n_rows = int(np.sqrt(images_to_show))
    n_cols = int(np.ceil(images_to_show / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(indices):
            ax.axis('off')
            continue

        image_index = indices[idx]

        img = images[image_index].reshape((112, 92))
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {labels[image_index]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def split_and_normalize(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
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
    print(X_train_orig[:5])
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_val_scaled = scaler.transform(X_val_orig)
    X_test_scaled = scaler.transform(X_test_orig)

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            scaler)


def plot_label_distribution(y_train, y_val, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, y, subset_name in zip(
            axes,
            [y_train, y_val, y_test],
            ["Train", "Validation", "Test"]):
        counts = pd.Series(y).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"{subset_name} Distribution")
        ax.set_xlabel("Person (label)")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


def run_pca(X_train, X_val, X_test):
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    variances = [0.95, 0.96, 0.97, 0.98, 0.99]
    for variance in variances:
        n_components_needed = np.searchsorted(cumulative, variance) + 1
        print(f"PCA: variance threshold = {variance}")
        print(f"Number of components needed = {n_components_needed}")
        print("\n")

    # Plot cumulative explained variance
    max_to_show = min(200, len(cumulative))
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, max_to_show + 1), cumulative[:max_to_show], marker='o')
    plt.axhline(0.95, linestyle='--', label=f"threshold={0.95}")
    plt.axhline(0.99, linestyle='--', label=f"threshold={0.99}", color='red')
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA – Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Fit PCA with set variance
    pca_final = PCA(n_components=0.99, random_state=RANDOM_STATE)
    pca_train = pca_final.fit_transform(X_train)
    pca_val = pca_final.transform(X_val)
    pca_test = pca_final.transform(X_test)

    # 2D projection just for visualization
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_train_2d = pca_2d.fit_transform(X_train)

    return pca_final, (pca_train, pca_val, pca_test), pca_train_2d


def plot_2d_embedding(features, labels, title):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(features[:, 0], features[:, 1],
                          c=labels, s=10, alpha=0.8)
    plt.colorbar(scatter, label='Label')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()


def build_autoencoder(input_dim, bottleneck_dim=100):
    autoencoder = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Dense(128, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Dense(bottleneck_dim, activation='leaky_relu', name='bottleneck', kernel_initializer="he_normal"),
        layers.Dense(128, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Dense(256, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Dense(input_dim, activation=None)
    ])

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    bottleneck_layer = autoencoder.get_layer('bottleneck').output
    encoder = models.Model(inputs=autoencoder.inputs, outputs=bottleneck_layer)

    return autoencoder, encoder


def train_autoencoder(autoencoder, X_train, X_val, epochs=20, batch_size=64):
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
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


def show_autoencoder_reconstructions(autoencoder, images, n_samples=10):
    indices = np.random.choice(images.shape[0], n_samples, replace=False)

    X_orig = images[indices]
    X_recon = autoencoder.predict(X_orig)

    n_rows = 2
    n_cols = n_samples

    plt.figure(figsize=(2 * n_cols, 4))

    # Original images
    for i, idx in enumerate(indices):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(X_orig[i].reshape((112, 92)), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

    # Reconstructed images
    for i, idx in enumerate(indices):
        ax = plt.subplot(n_rows, n_cols, n_cols + i + 1)
        ax.imshow(X_recon[i].reshape((112, 92)), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    images, labels = load_data()
    show_sample_images(images, labels, 12)
    (X_train, X_val, X_test, y_train, y_val, y_test, _) = split_and_normalize(images, labels)

    plot_label_distribution(y_train, y_val, y_test)

    _, (X_pca_train, X_pca_val, X_pca_test), pca_train_2d = run_pca(X_train, X_val, X_test)
    plot_2d_embedding(pca_train_2d, y_train, "PCA for only 2 components")

    autoencoder, encoder = build_autoencoder(X_train.shape[1], 64)
    train_autoencoder(autoencoder, X_train, X_val, 50, 64)
    show_autoencoder_reconstructions(autoencoder, X_test, 10)

    autoencoder, encoder = build_autoencoder(X_train.shape[1], 2)
    train_autoencoder(autoencoder, X_train, X_val, 20, 64)
    show_autoencoder_reconstructions(autoencoder, X_test, 10)

    autoencoder_2d_train = encoder.predict(X_train)
    plot_2d_embedding(autoencoder_2d_train, y_train, "Autoencoder for only 2 components")


main()
