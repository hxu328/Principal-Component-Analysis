from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    s = np.zeros((len(dataset[0]), len(dataset[0])))  # create an empty NumPy array
    for point in dataset:                             # traverse array elements
        s += np.dot(np.transpose([point]), [point])
    s = s * 1 / (len(dataset) - 1)
    return s


def get_eig(S, m):
    values, vectors = eigh(S, eigvals=[len(S) - m, len(S) - 1])    # get m Largest Eigenvalues/Eigenvectors
    values = np.flip(values)                                       # convert ascending to decreasing
    values = np.diag(values)                                       # convert to diagonal matrix
    vectors = np.flip(vectors, 1)                                  # rearrange corresponding vectors
    return values, vectors


def get_eig_perc(S, perc):
    values, vectors = eigh(S)
    vectors = np.flip(vectors, 1)
    sum_of_value = np.sum(values)
    filters = []
    for element in values:
        if element > (sum_of_value * perc):
            filters.append(True)
        else:
            filters.append(False)
    values = values[filters]
    values = np.sort(values)
    values = np.flip(values)
    vectors = vectors[:, 0:len(values)]
    values = np.diag(values)
    return values, vectors


def project_image(img, U):
    num_rows, num_cols = U.shape
    x_pro = np.zeros(num_rows)
    i = 0
    while i < num_cols:
        x_pro += np.dot(U[:, i], img) * U[:, i]
        i += 1
    return x_pro


def display_image(orig, proj):
    orig = orig.reshape((32, 32), order='F')
    proj = proj.reshape((32, 32), order='F')
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    img_orig = ax1.imshow(orig, aspect='equal')
    img_proj = ax2.imshow(proj, aspect='equal')

    f.colorbar(img_orig, ax=ax1, shrink=0.5)
    f.colorbar(img_proj, ax=ax2, shrink=0.5)
    plt.show()
