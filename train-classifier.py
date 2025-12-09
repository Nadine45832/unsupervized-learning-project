import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras import layers, models, callbacks, Sequential, optimizers

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

    return np.vstack(all_images), np.array(all_labels, dtype=np.int32)


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


def split_and_pixel_normalize(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
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

    print(X_train_orig[:5])
    X_train_scaled = X_train_orig / 255.0
    X_val_scaled = X_val_orig / 255.0
    X_test_scaled = X_test_orig / 255.0

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test)


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
        layers.Dense(512, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dense(256, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dense(bottleneck_dim, activation='leaky_relu', name='bottleneck', kernel_initializer="he_normal"),
        layers.Dense(256, activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Dense(512, activation='leaky_relu', kernel_initializer="he_normal"),
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

def compute_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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


def claster_with_kmeans(features):
    results = {"k": [], "inertia": [], "silhouette": []}

    for k in range(3, 41):
        kmean = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmean.fit_predict(features)
        inertia = kmean.inertia_
        sil = silhouette_score(features, labels, metric='euclidean')
        results["k"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(sil)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results["k"], results["inertia"], marker='o')
    plt.xlabel("k clusters")
    plt.ylabel("Inertia")
    plt.title("KMeans: Inertia vs k")

    plt.subplot(1, 2, 2)
    plt.plot(results["k"], results["silhouette"], marker='o')
    plt.xlabel("k clusters")
    plt.ylabel("Silhouette score")
    plt.title("KMeans: Silhouette vs k")
    plt.tight_layout()
    plt.show()

    kmean = KMeans(n_clusters=20, random_state=RANDOM_STATE, n_init=10)
    return kmean.fit_predict(features)


def claster_with_dbscan(features, title):
    eps_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    min_samples_values = range(5, 21)
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = db.fit_predict(features)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)

            # skip results when everything is outliers and 1 or 2 cluster
            # or if 40% of points are outliers
            if n_clusters < 3 or noise_ratio > 0.4:
                continue

            real_clusters = labels != -1
            silhouette = silhouette_score(
                features[real_clusters], labels[real_clusters])
            noise_ratio = n_noise / len(labels)
            results.append({
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "eps": eps,
                "min_samples": min_samples,
                "silhouette": silhouette,
                "noise_ratio": noise_ratio
            })

    for res in sorted(results, key=lambda r: r['silhouette']):
        print(f"\nDBScan results or {title}:")
        print(f"  eps: {res['eps']}")
        print(f"  min_samples: {res['min_samples']}")
        print(f"  Number of clusters: {res['n_clusters']}")
        print(f"  Noise points: {res['n_noise']} ({res['noise_ratio']:.2%})")
        print(f"  Silhouette Score: {res['silhouette']}")

    return results


def visualize_dbscan(results):

    # Sort by silhouette score
    sorted_results = sorted(results, key=lambda x: x['silhouette'], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 12))

    # Silhouette scores
    ax = axes[0]
    labels_plot = [f"eps={r['eps']}\nmin={r['min_samples']}" for r in sorted_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_results)))
    ax.barh(range(len(sorted_results)), [r['silhouette'] for r in sorted_results], color=colors)
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(labels_plot)
    ax.set_xlabel('Silhouette Score')
    ax.set_title('Top Parameter Combinations by Silhouette Score')
    ax.grid(True, alpha=0.3)

    # Number of clusters vs noise ratio
    ax = axes[1]
    scatter = ax.scatter([r['n_clusters'] for r in results],
                        [r['noise_ratio'] for r in results],
                        c=[r['silhouette'] for r in results],
                        s=100, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Noise Ratio')
    ax.set_title('Clusters vs Noise Ratio (colored by Silhouette)')
    plt.colorbar(scatter, ax=ax, label='Silhouette Score')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def claster_with_gmm(images):
    results = {"n_components": [], "bic": [], "aic": []}
    covariance_types = ['full', 'tied', 'diag', 'spherical']

    features = images.astype(np.float64)
    results = []

    for c_type in covariance_types:
        r = {"n_components": [], "bic": [], "aic": [], "covariance_type": c_type}
        results.append(r)

        for n in range(10, 30):
            gm = GaussianMixture(n_components=n, covariance_type=c_type, random_state=RANDOM_STATE, max_iter=200)
            gm.fit(features)
            labels = gm.predict(features)
            bic = gm.bic(features)
            aic = gm.aic(features)

            r["n_components"].append(n)
            r["bic"].append(bic)
            r["aic"].append(aic)

    for r in results:
        c_type = r["covariance_type"]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(r["n_components"], r["bic"], marker='o')
        plt.xlabel("n_components")
        plt.ylabel("BIC")
        plt.title(f"GMM  ({c_type}): BIC vs n_components")

        plt.subplot(1, 2, 2)
        plt.plot(r["n_components"], r["aic"], marker='o')
        plt.xlabel("n_components")
        plt.ylabel("AIC")
        plt.title(f"GMM ({c_type}): AIC vs n_components")

        plt.tight_layout()
        plt.show()


def build_cnn_classifier(input_dim, num_classes):
    model = Sequential([
        layers.Input(shape=(112, 92, 1)),
        layers.Conv2D(128, (3, 3), activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Conv2D(128, (3, 3), activation='leaky_relu', kernel_initializer="he_normal"),
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation='leaky_relu', kernel_initializer="he_normal"),
        layers.Conv2D(64, (3, 3), activation='leaky_relu', kernel_initializer="he_normal"),
        layers.MaxPooling2D((3, 3)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(64, activation='leaky_relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train_and_evaluate_cnn_classifier(Z_train, y_train,
                                      Z_val, y_val,
                                      Z_test, y_test,
                                      num_classes):
    
    Z_train_reshaped = Z_train.reshape(-1, 112, 92, 1).astype(np.float32).copy()
    Z_val_reshaped = Z_val.reshape(-1, 112, 92, 1).astype(np.float32).copy()
    Z_test_reshaped = Z_test.reshape(-1, 112, 92, 1).astype(np.float32).copy()

    model = build_cnn_classifier(Z_train_reshaped, num_classes)

    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        Z_train_reshaped, y_train,
        validation_data=(Z_val_reshaped, y_val),
        epochs=20,
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
    test_loss, test_acc = model.evaluate(Z_test_reshaped, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

    # Confusion matrix & classification report
    y_pred_probs = model.predict(Z_test_reshaped)
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

def build_mlp_classifier(input_dim, num_classes):
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


def train_and_evaluate_ann_classifier(Z_train, y_train,
                                  Z_val, y_val,
                                  Z_test, y_test,
                                  title):
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

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"NN Classifier – Accuracy {title}")
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
    plt.title(f"NN Classifier – Loss {title}")
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
    plt.title(f"Confusion Matrix (Test) {title}")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    return model


def main():
    images, labels = load_data()
    show_sample_images(images, labels, 12)
    (X_train, X_val, X_test, y_train, y_val, y_test) = split_and_pixel_normalize(images, labels)

    plot_label_distribution(y_train, y_val, y_test)

    # Feature reduction
    _, (X_pca_train, X_pca_val, X_pca_test), pca_train_2d = run_pca(X_train, X_val, X_test)
    plot_2d_embedding(pca_train_2d, y_train, "PCA for only 2 components")

    autoencoder, encoder = build_autoencoder(X_train.shape[1], 100)
    train_autoencoder(autoencoder, X_train, X_val, 50, 64)
    show_autoencoder_reconstructions(autoencoder, X_test, 10)
    autoencoder_train = encoder.predict(X_train)
    autoencoder_normalized_train = normalize(autoencoder_train, norm="l2")

    autoencoder_2d, encoder_2d = build_autoencoder(X_train.shape[1], 2)
    train_autoencoder(autoencoder_2d, X_train, X_val, 20, 64)
    show_autoencoder_reconstructions(autoencoder_2d, X_test, 10)
    autoencoder_2d_train = encoder_2d.predict(X_train)
    plot_2d_embedding(autoencoder_2d_train, y_train, "Autoencoder for only 2 components")

    # Clustering
    pca_labels = claster_with_kmeans(X_pca_train)
    print("KMeans (PCA) purity: ", compute_purity(y_train, pca_labels))
    autoencoder_labels = claster_with_kmeans(autoencoder_normalized_train)
    print("KMeans (Autoencoder) purity: ", compute_purity(y_train, autoencoder_labels))

    results = claster_with_dbscan(X_pca_train, "PCA")
    visualize_dbscan(results)
    db = DBSCAN(eps=0.25, min_samples=5, metric='cosine')
    print("DBScan (PCA) purity: ", compute_purity(y_train, db.fit_predict(X_pca_train)))

    results = claster_with_dbscan(autoencoder_normalized_train, "Autoencoder")
    visualize_dbscan(results)
    db = DBSCAN(eps=0.1, min_samples=5, metric='cosine')
    print("DBScan (Autoencoder) purity: ", compute_purity(y_train, db.fit_predict(autoencoder_normalized_train)))

    claster_with_gmm(X_pca_train)
    features = X_pca_train.astype(np.float64)
    gm = GaussianMixture(n_components=24, covariance_type="diag", random_state=RANDOM_STATE, max_iter=200)
    gm.fit(features)
    purity = compute_purity(y_train, gm.predict(features))
    print(f"GMM (PCA) AIC {gm.aic(features)} BIC: {gm.bic(features)} Purity: {purity}")

    claster_with_gmm(autoencoder_normalized_train)
    gm = GaussianMixture(n_components=24, covariance_type="tied", random_state=RANDOM_STATE, max_iter=200)
    features = autoencoder_normalized_train.astype(np.float64)
    gm.fit(features)
    purity = compute_purity(y_train, gm.predict(features))
    print(f"GMM (Autoencoder) AIC {gm.aic(features)} BIC: {gm.bic(features)} Purity: {purity}")


    # Classification with ANN on features
    classifier_model = train_and_evaluate_ann_classifier(
        X_pca_train, y_train,
        X_pca_val, y_val,
        X_pca_test, y_test,
        "PCA"
    )

    classifier_model = train_and_evaluate_ann_classifier(
        encoder.predict(X_train), y_train,
        encoder.predict(X_val), y_val,
        encoder.predict(X_test), y_test,
        "Autoencoder"
    )

    (X_train_2, X_val_2, X_test_2, y_train_2, y_val_2, y_test_2) = split_and_pixel_normalize(images, labels)
    classifier_model = train_and_evaluate_cnn_classifier(
        X_train_2, y_train_2,
        X_val_2, y_val_2,
        X_test_2, y_test_2,
        num_classes=len(set(labels))
    )


main()
