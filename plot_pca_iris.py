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

# Load Iris dataset
iris = load_iris(as_frame=True)

# Tampilkan keys data untuk informasi
print("Data keys:", iris.keys())

# Tambahkan kolom target nama kelas ke dataframe
iris.frame["target"] = iris.target_names[iris.target]

# Visualisasi pairplot untuk semua fitur
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
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Tambah legend kelas
legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Kelas",
)
ax.add_artist(legend1)

plt.show()

# Catatan singkat
print("""
PCA berhasil mengurangi dimensi dari 4 fitur ke 3 komponen utama.
Setosa dapat dipisahkan dengan jelas,
sementara Versicolor dan Virginica masih sedikit tumpang tindih.
""")
