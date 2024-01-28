import pandas as pd
import os
import argparse
import json


def combine_json_to_csv(directory):
    json_files = [
        pos_json for pos_json in os.listdir(directory) if pos_json.endswith(".json")
    ]
    combined_df = pd.DataFrame()

    for index, file in enumerate(json_files):
        file_path = os.path.join(directory, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            # Ensure data is a list of records
            if isinstance(data, dict):
                data = [data]
            temp_df = pd.DataFrame(data)
            combined_df = pd.concat(
                [combined_df, temp_df], ignore_index=True, sort=False
            )

    # Fill missing values with NaN
    combined_df = combined_df.where(pd.notnull(combined_df), None)

    # Save to CSV
    combined_df.to_csv(os.path.join(directory, "combined_output.csv"), index=False)
    print(f"Combined CSV created at {os.path.join(directory, 'combined_output.csv')}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine JSON files into a single CSV."
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing JSON files"
    )
    args = parser.parse_args()

    combine_json_to_csv(args.directory)


if __name__ == "__main__":
    main()
