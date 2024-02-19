import os
from posixpath import basename
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("POSTGRES_DB_CREDENTIALS")
engine = create_engine(url)  # type: ignore

txtfile = "../data/m&a/lumen/Lumen_Incumbent_Local_Exchange_Carrier_Business_Apollo_Global_Management_LLC_7_500m_Announce_20210803_merger_agree_20210804.pdf.txt"
prompt_path = "./"


def get_new_doc_id(filename: str):
    query = "SELECT MAX(id) FROM experiments.documents;"
    try:
        doc_id = pd.read_sql(query, engine).values[0][0]
        new_doc_id = int(doc_id) + 1
    except Exception:
        new_doc_id = 1
        pd.DataFrame({"id": [1], "filename": [filename]}).to_sql(
            "documents", engine, index=False, if_exists="append", schema="experiments"
        )

    # Use parameterized query for INSERT
    insert_query = text(
        "INSERT INTO experiments.documents (id, filename) VALUES (:id, :filename)"
    )
    with engine.connect() as conn:
        conn.execute(insert_query, {"id": new_doc_id, "filename": filename})
        conn.commit()

    return new_doc_id


def process_and_insert_txt_to_db(filename: str) -> int:
    """
    Reads a text file, processes its content into a DataFrame, and inserts it into a PostgreSQL database.

    Parameters:
    - filename: str - Path to the text file to be processed.
    - engine: sqlalchemy.engine.Engine - SQLAlchemy engine object for database connection.
    - table_name: str - Name of the table where data will be inserted.
    - schema_name: str - Name of the database schema. Defaults to "experiments".
    """
    # Read the file and create a DataFrame
    with open(filename, "r") as file:
        txt = file.read()
        df_pdf = pd.DataFrame(txt.split("\n"), columns=["line"])

    # Add columns for file ID and filename
    new_doc_id = get_new_doc_id(filename)
    df_pdf["document_id"] = new_doc_id

    # Extract the tag from each line, looking for first "/>" and extracting the string before it
    df_pdf["tag"] = df_pdf["line"].apply(
        lambda x: x.split("/>")[0] + "/>" if "/>" in x else None
    )
    df_pdf["line"] = df_pdf["line"].apply(
        lambda x: x.split("/>")[1] if "/>" in x else x
    )

    # Insert the DataFrame into the database
    df_pdf.to_sql(
        "documents_tags", engine, index=False, if_exists="append", schema="experiments"
    )

    print(df_pdf.head())
    return new_doc_id


def insert_in_out_table(prompt_path: str, document_id: int):
    # Read all the files that contain .in.txt and .out.json
    # write sql to table and associate them with the document_id
    in_out_files = [
        f
        for f in os.listdir(prompt_path)
        if f.endswith(".in.txt") or f.endswith(".out.json")
    ]
    for file in in_out_files:
        if file.endswith(".in.txt"):
            with open(prompt_path + file, "r") as f:
                content = f.read()
                insert_query = text(
                    "INSERT INTO experiments.in_out_files (document_id, file_type, file_content) VALUES (:document_id, :file_type, :file_content)"
                )
                with engine.connect() as conn:
                    conn.execute(
                        insert_query,
                        {
                            "document_id": document_id,
                            "file_type": "in",
                            "file_content": content,
                        },
                    )
                    conn.commit()
        elif file.endswith(".out.json"):
            with open(prompt_path + file, "r") as f:
                content = f.read()
                insert_query = text(
                    "INSERT INTO experiments.in_out_files (document_id, file_type, file_content) VALUES (:document_id, :file_type, :file_content)"
                )
                with engine.connect() as conn:
                    conn.execute(
                        insert_query,
                        {
                            "document_id": document_id,
                            "file_type": "out",
                            "file_content": content,
                        },
                    )
                    conn.commit()


def init_schema(schema_name="experiments"):
    """
    Initializes the schema in the database if it does not exist.

    Parameters:
    - engine: sqlalchemy.engine.Engine - SQLAlchemy engine object for database connection.
    - schema_name: str - The name of the schema to initialize.
    """
    create_schema_statement = text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
    with engine.connect() as conn:
        conn.execute(create_schema_statement)
        print(f"Schema '{schema_name}' is ensured.")

    documents_table_sql = text(
        """
        CREATE TABLE IF NOT EXISTS experiments.documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR NOT NULL
        );
    """
    )

    documents_tags_table_sql = text(
        """
        CREATE TABLE IF NOT EXISTS experiments.documents_tags (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES experiments.documents(id),
            line TEXT NOT NULL,
            tag TEXT
        );
    """
    )

    inout_files_table_sql = text(
        """
        CREATE TABLE IF NOT EXISTS experiments.in_out_files (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES experiments.documents(id),
            file_type VARCHAR(50) NOT NULL,
            file_content TEXT NOT NULL
        );
    """
    )

    with engine.connect() as conn:
        conn.execute(documents_table_sql)
        conn.execute(documents_tags_table_sql)
        conn.execute(inout_files_table_sql)
        conn.commit()
        print("Tables are ensured in the database.")


def insert_metrics(id: int):
    df = pd.read_csv("metrics.csv")
    df["document_id"] = id
    df.to_sql("metrics", engine, index=False, if_exists="append", schema="experiments")


def insert_pass1_results(id: int):
    df = pd.read_csv("pass2.csv")
    df["document_id"] = id
    df.to_sql(
        "pass1_results", engine, index=False, if_exists="append", schema="experiments"
    )


if __name__ == "__main__":
    # init_schema()
    # id = process_and_insert_txt_to_db(txtfile)
    # insert_in_out_table(prompt_path, id)
    id = 4
    # insert_metrics(id)
    insert_pass1_results(id)
    print("Data inserted successfully.")
