

from sklearn.datasets import fetch_olivetti_faces
import numpy as np

# Download Olivetti faces dataset
olivetti = fetch_olivetti_faces()
x = olivetti.images
y = olivetti.target

# Print info on shapes and reshape where necessary
print("Original x shape:", x.shape)
X = x.reshape((400, 4096))
print("New x shape:", X.shape)
print("y shape", y.shape)

# Save the numpy arrays
np.savetxt("D:\Google Drive\Programming\Projects\Facial Image Recognition using CNNs - Olivetti Faces\olivetti_x.csv", X, delimiter = ",")
np.savetxt("D:\Google Drive\Programming\Projects\Facial Image Recognition using CNNs - Olivetti Faces\olivetti_y.csv", y, delimiter = ",", fmt = '%d')

print("\nDownloading and reshaping done!")


###
### Output:
###
#
# Original x shape: (400, 64, 64)
# New x shape: (400, 4096)
# y shape (400,)
#
# Downloading and reshaping done!