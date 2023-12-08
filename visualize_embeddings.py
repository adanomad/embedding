# Visualizing a 1024-dimensional vector in a 3D plot is a challenging task because it
# involves reducing the dimensions from 1024 to just 3. This can be achieved using 
# dimensionality reduction techniques such as Principal Component Analysis (PCA) 
# or t-Distributed Stochastic Neighbor Embedding (t-SNE). These techniques transform the 
# high-dimensional data into a lower-dimensional space (in this case, 3D) while trying to
#  reserve as much of the significant information as possible.
# 
# This script that visualizes vectors from a CSV file. 
# The script will read the specified column from the CSV file, 
# where each row contains a string representation of a vector (e.g., "[1,2,...,3]"). 
# It will then convert these strings into actual vectors and visualize them using PCA 
# for dimensionality reduction, as the vector lengths could vary.
#
# install the pandas numpy matplotlib scikit-learn packages
# python visualize_embeddings.py data.csv vector_column


import pandas as pd
# import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import argparse
import ast

def read_vectors_from_csv(filename, column_name):
    df = pd.read_csv(filename)
    return df[column_name].apply(ast.literal_eval).tolist()

def check_consistent_length(vectors):
    length = len(vectors[0])
    return all(len(v) == length for v in vectors)

def main(filename, column_name):
    vectors = read_vectors_from_csv(filename, column_name)

    if not check_consistent_length(vectors):
        raise ValueError("Vectors are not of consistent lengths.")

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=3)
    vectors_reduced = pca.fit_transform(vectors)

    # 3D Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], vectors_reduced[:, 2])

    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize vectors from a CSV file.")
    parser.add_argument('filename', type=str, help='Path to the CSV file.')
    parser.add_argument('column_name', type=str, help='Name of the column containing vectors.')
    args = parser.parse_args()

    main(args.filename, args.column_name)
