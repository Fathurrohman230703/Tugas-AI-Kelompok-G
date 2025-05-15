# iris_pca_project.ipynb

# Source: Scikit-learn example code (BSD-3-Clause License)
# Original authors: Scikit-learn developers
# Modified by: Kelompok G, for academic purposes only
# License: BSD-3-Clause

from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# Load Iris dataset
iris = load_iris(as_frame=True)

# Tambahkan kolom target nama kelas ke dataframe
iris.frame["target"] = iris.target_names[iris.target]

# Pairplot untuk eksplorasi awal
sns.pairplot(iris.frame, hue="target")
plt.suptitle("Pairplot fitur dataset Iris", y=1.02)
plt.show()

# PCA: reduksi dari 4 dimensi ke 3 dimensi
X_reduced = PCA(n_components=3).fit_transform(iris.data)

# Visualisasi 3D hasil PCA
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set(
    title="PCA - Proyeksi ke 3 Dimensi",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)
legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Kelas",
)
ax.add_artist(legend1)
plt.show()

# --------- MODEL SUPERVISED LEARNING ---------
# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Gunakan model K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi Model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("\nAkurasi:", accuracy_score(y_test, y_pred))

# Visualisasi Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=iris.target_names)
plt.title("Confusion Matrix - KNN (k=3)")
plt.show()
