import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

def generate_points():
    """ Generate random points. """
    centers = []
    centers.append([[ 0.75,-0.75]])
    centers.append([[-0.75,-0.75]])
    centers.append([[ 0.75, 0.75]])
    datas = [None] * len(centers)
    labels_true = [None] * len(centers)
    datas[0], labels_true[0] = make_blobs(n_samples=40, centers=centers[0], cluster_std=0.4)
    datas[1], labels_true[1] = make_blobs(n_samples=15, centers=centers[1], cluster_std=0.10)
    datas[2], labels_true[2] = make_blobs(n_samples=25, centers=centers[2], cluster_std=0.20)
    print(datas[0])
    print(len(datas))
    for i in range(len(datas)-1):
        data = np.append(datas[0], datas[i+1], axis=0)
    data = pd.DataFrame(data, columns=['x','y'])
    filee = open("./datasets/test.txt", "w", newline = '')
    data.to_csv(filee, sep = ',', index = False)

def main():
    generate_points()
if __name__ == "__main__":
    main()
