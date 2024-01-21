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
from sklearn.decomposition import PCA
import plotly.offline as pyo
import plotly.graph_objs as go
import argparse
import ast


def read_vectors_and_names_from_csv(filename, name_column, vector_column):
    df = pd.read_csv(filename)
    names = df[name_column].tolist()
    vectors = df[vector_column].apply(ast.literal_eval).tolist()
    return names, vectors


def check_consistent_length(vectors):
    length = len(vectors[0])
    return all(len(v) == length for v in vectors)


def main(filename, name_column, vector_column):
    names, vectors = read_vectors_and_names_from_csv(
        filename, name_column, vector_column
    )

    if not check_consistent_length(vectors):
        raise ValueError("Vectors are not of consistent lengths.")

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=3)
    vectors_reduced = pca.fit_transform(vectors)

    # Create a DataFrame for Plotly
    df_plot = pd.DataFrame(vectors_reduced, columns=["PCA1", "PCA2", "PCA3"])
    df_plot["File Name"] = names

    # 3D Plotting with Plotly
    trace = go.Scatter3d(
        x=df_plot["PCA1"],
        y=df_plot["PCA2"],
        z=df_plot["PCA3"],
        text=df_plot["File Name"],
        mode="markers",
        marker=dict(size=5),
    )
    layout = go.Layout(title="3D Scatter plot")
    fig = go.Figure(data=[trace], layout=layout)

    pyo.plot(fig, filename=f"{filename}.html")
    print(f"Plot saved to {filename}.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize vectors from a CSV file using Plotly."
    )
    parser.add_argument("filename", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "name_column", type=str, help="Name of the column containing file names."
    )
    parser.add_argument(
        "vector_column", type=str, help="Name of the column containing vectors."
    )
    args = parser.parse_args()

    main(args.filename, args.name_column, args.vector_column)
