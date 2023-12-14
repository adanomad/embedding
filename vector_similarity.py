import csv
import numpy as np

def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]
    return rows

def is_size_similar(size1, size2):
    return abs(size1 - size2) / ((size1 + size2) / 2) < 0.02

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def process_csv(file_path):
    rows = read_csv(file_path)
    for i in range(1, len(rows)):
        size_current = int(rows[i]['size'])
        size_previous = int(rows[i-1]['size'])

        if is_size_similar(size_previous, size_current):
            embedding_current = [int(x) for x in rows[i]['embedding'].strip('[]').split(',')]
            embedding_previous = [int(x) for x in rows[i-1]['embedding'].strip('[]').split(',')]
            distance = cosine_similarity(np.array(embedding_current), np.array(embedding_previous))

            if distance > 0.98:
                print(f"{rows[i]['image']} similar to {rows[i-1]['image']} with distance {distance * 100:.3f}%")

process_csv("ss.csv")
