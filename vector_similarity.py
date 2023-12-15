import csv
import numpy as np
import argparse
import os
import shutil

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

def move_files(suggestions, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_name in suggestions:
        shutil.move(file_name, os.path.join(target_dir, os.path.basename(file_name)))

def process_csv(file_path, threshold):
    rows = read_csv(file_path)
    suggestions = []

    for i in range(1, len(rows)):
        size_current = int(rows[i]['size'])
        size_previous = int(rows[i-1]['size'])

        if is_size_similar(size_previous, size_current):
            embedding_current = [int(x) for x in rows[i]['embedding'].strip('[]').split(',')]
            embedding_previous = [int(x) for x in rows[i-1]['embedding'].strip('[]').split(',')]
            distance = cosine_similarity(np.array(embedding_current), np.array(embedding_previous))

            if distance > threshold:
                print(f"{rows[i]['image']} similar to {rows[i-1]['image']} with distance {distance * 100:.3f}%")
                suggestions.append(rows[i]['image'])

    if suggestions:
        target_dir = os.path.join(os.path.dirname(file_path), 'similar')
        move_files(suggestions, target_dir)

def main():
    parser = argparse.ArgumentParser(description="Process a CSV file and output suggestions for deletion based on file size and embedding similarity.")
    parser.add_argument('file_path', type=str, help="Path to the CSV file")
    parser.add_argument('--threshold', type=float, help="Threshold for similarity", default=1.00)
    args = parser.parse_args()

    process_csv(args.file_path, args.threshold)

if __name__ == "__main__":
    main()
