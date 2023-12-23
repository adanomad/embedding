// src/EmbeddingSearch.jsx
import { useState } from "react";
import axios from "axios";
import { useLazyQuery } from "@apollo/client";
import {
  Anchor,
  Text,
  TextInput,
  Button,
  Card,
  Image,
  Pagination,
} from "@mantine/core";
import { gql } from "@apollo/client";
import { usePagination } from "@mantine/hooks";

const GET_CLOSEST_FILE_EMBEDDING = gql`
  query GetClosestFileEmbedding($embedding: [Float!]!) {
    Get {
      FileEmbedding(
        nearVector: { vector: $embedding, distance: 0.80 }
        limit: 3
      ) {
        fileName
        _additional {
          distance
        }
      }
    }
  }
`;

const EmbeddingSearch = () => {
  const [inputText, setInputText] = useState("");
  const [embedding, setEmbedding] = useState(null);
  const [getClosestFileEmbedding, { data, loading, error }] = useLazyQuery(
    GET_CLOSEST_FILE_EMBEDDING
  );
  const [page, onChange] = useState(1);
  const pagination = usePagination({ total: 10, page, onChange });

  const handleSearch = async () => {
    try {
      const response = await axios.post("http://glassbox.ds:5000/embed", {
        text: inputText,
      });
      setEmbedding(response.data.embedding);
      getClosestFileEmbedding({
        variables: { embedding: response.data.embedding, limit: 10, offset: 1 },
      });
    } catch (error) {
      console.error("Error fetching embedding:", error);
    }
  };

  const handlePageChange = (page: number) => {
    pagination.setPage(page);
    const offset = (page - 1) * 5;
    if (embedding) {
      getClosestFileEmbedding({ variables: { embedding, offset } });
    }
  };

  const flaskEndpoint = "http://glassbox.ds:5000/image?fileName="; // Update with your Flask app's URL

  return (
    <Card shadow="sm" padding="lg">
      <TextInput
        placeholder="Enter text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            handleSearch();
          }
        }}
      />
      <Button onClick={handleSearch}>Get Embedding and Search</Button>

      {loading && <p>Loading...</p>}
      {error && <p>Error: {error.message}</p>}
      {data && (
        <Card>
          {data.Get.FileEmbedding.map(
            (
              item: { fileName: string; _additional: { distance: number } },
              index: number
            ) => (
              <Card key={index}>
                <Anchor href={`${flaskEndpoint}${item.fileName}`}>
                  {index} File Name: {item.fileName}
                </Anchor>
                <Text>Distance: {item._additional.distance.toFixed(2)}</Text>
                <Image
                  src={`${flaskEndpoint}${item.fileName}`}
                  alt={item.fileName}
                  width={400}
                  height={400}
                />
              </Card>
            )
          )}
          <Pagination total={10} onChange={handlePageChange} />
        </Card>
      )}
    </Card>
  );
};

export default EmbeddingSearch;
