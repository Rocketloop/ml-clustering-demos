import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

def generate_points():
    """ Generate random points. """
    centers_ld = [[ 0.75,-0.75]]
    centers_hd = [[-0.75,-0.75]]
    centers_nd = [[ 0.75, 0.75]]
    data, labels_true = make_blobs(n_samples= 40, centers=centers_ld, cluster_std=0.4)
    data2, labels_true = make_blobs(n_samples=15, centers=centers_hd, cluster_std=0.10)
    data3, labels_true = make_blobs(n_samples=25, centers=centers_nd, cluster_std=0.20)
    data = np.append(data, data2, axis=0)
    data = np.append(data, data3, axis=0)
    data = pd.DataFrame(data, columns=['x','y'])
    data2 = data
    filee = open("./datasets/test.txt", "w", newline = '')
    data.to_csv(filee, sep = ',', index = False)

def main():
    generate_points()
if __name__ == "__main__":
    main()
