import os
import hashlib
from clickhouse_driver import Client
import numpy as np

# Need to setup first in the clickhouse db:
# CREATE TABLE vector_storage (
#     embedding_vector Array(UInt8),
#     md5_hash FixedString(32),
#     file_name String
# ) ENGINE = MergeTree()
# ORDER BY md5_hash;


# Environment setup
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
# Retrieve configuration from environment variables
host = os.getenv("CH_HOST", "localhost")
port = os.getenv("CH_PORT", "9000")
user = os.getenv("CH_USER", "default")
password = os.getenv("CH_PASSWORD", "")
database = os.getenv("CH_DATABASE", "default")
ch_client = Client(
    host=host, port=port, user=user, password=password, database=database
)


def get_md5_hash(file_name: str) -> str:
    """Generate MD5 hash for a given file."""
    with open(file_name, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def insert_into_db(
    client: Client, embedding_vector: np.ndarray, file_name: str
) -> None:
    """Insert a new row into the ClickHouse database."""
    md5_hash = get_md5_hash(file_name)
    client.execute(
        "INSERT INTO vector_storage (embedding_vector, md5_hash, file_name) VALUES",
        [(embedding_vector, md5_hash, file_name)],
    )


def main():
    # Add entry to vector db
    insert_into_db(ch_client, [1, 2, 3, 4, 5], "sample.jpg")


if __name__ == "__main__":
    main()

# Should look like this afterwards:
# clickhouse-client --user <user> --password <pass> --database <db>
# SELECT * FROM vector_storage
# Query id: be04cd59-3052-43af-af37-2b9dd67ede42
# ┌─embedding_vector─┬─md5_hash─────────────────────────┬─file_name──┐
# │ [1,2,3,4,5]      │ 09304574e6bcd353beeb3514dfbe0e94 │ sample.jpg │
# └──────────────────┴──────────────────────────────────┴────────────┘
# 1 row in set. Elapsed: 0.011 sec.
