# README

## Description 
The embedding.py file takes an image and calculates a 1024 uint8 vector result as the embedding.

This vector can be added to a database and used to compare with other embeddings to find the closest match. We use ClickHouseDB for this purpose.

## Database preparation
To create a database and user in ClickHouseDB, run the following commands:

```
clickhouse-client

CREATE DATABASE mydatabase;

CREATE USER myuser IDENTIFIED BY 'mypassword';

GRANT ALL ON mydatabase.* TO myuser;

QUIT;
```
Now you can login to ClickHouseDB with the following command:

```
clickhouse-client --user myuser --password mypassword --database mydatabase
```

Create a table in ClickHouseDB with the following command:

```
CREATE TABLE vector_storage (
    embedding_vector Array(UInt8),
    md5_hash FixedString(32),
    file_name String
) ENGINE = MergeTree()
ORDER BY md5_hash;
```
