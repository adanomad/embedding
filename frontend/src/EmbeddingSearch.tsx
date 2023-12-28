// src/EmbeddingSearch.jsx
import { useState } from "react";
import axios from "axios";
import {
  Anchor,
  Text,
  TextInput,
  Button,
  Card,
  Image,
  Pagination,
} from "@mantine/core";
import { usePagination } from "@mantine/hooks";
import { client } from "./weaviateClient";

interface SearchResult {
  fileName: string;
  _additional: { distance: number };
}

const EmbeddingSearch = () => {
  const [inputText, setInputText] = useState("");
  const [embedding, setEmbedding] = useState<number[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activePage, setActivePage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  // setTotalPages doesn't work because we don't know the total number of results - need estimate
  const limit = 5;

  const fetchSearchResults = async (embedding: number[], page: number) => {
    try {
      const offset = (page - 1) * limit;
      const results = await client.graphql
        .get()
        .withClassName("FileEmbedding")
        .withNearVector({ vector: embedding }) //distance: 0.68
        .withLimit(limit)
        .withOffset(offset)
        .withFields("fileName _additional { distance }")
        // .withAutocut(1)
        .do();

      // Reversing the order of search results
      const reversedResults = results.data.Get.FileEmbedding.slice().reverse();
      setSearchResults(reversedResults);

      // Calculate total pages if you have that information
      // setTotalPages(calculatedTotalPages);
    } catch (err) {
      console.error("Error fetching embedding:", err);
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    try {
      const embedResponse = await axios.post("http://glassbox.ds:5000/embed", {
        text: inputText,
      });
      setEmbedding(embedResponse.data.embedding);
      await fetchSearchResults(embedResponse.data.embedding, 1);
    } catch (err) {
      console.error("Error fetching embedding:", err);
      setError(err instanceof Error ? err.message : String(err));
    }
    setLoading(false);
  };

  const handlePageChange = (page: number) => {
    setActivePage(page);
    if (embedding.length > 0) {
      fetchSearchResults(embedding, page);
    } else {
      console.error("No embedding found");
    }
  };

  const flaskEndpoint = "http://glassbox.ds:5000/image?fileName=";

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
      {searchResults && (
        <Card>
          {searchResults.map(
            (
              item: { fileName: string; _additional: { distance: number } },
              index: number
            ) => (
              <Card key={index}>
                <Anchor href={`${flaskEndpoint}${item.fileName}`}>
                  {index} File Name: {item.fileName}
                </Anchor>
                <Text>Distance: {item._additional.distance.toFixed(4)}</Text>
                <Image
                  src={`${flaskEndpoint}${item.fileName}`}
                  alt={item.fileName}
                  width={400}
                  height={400}
                />
              </Card>
            )
          )}
          <Pagination
            value={activePage}
            total={totalPages}
            onChange={handlePageChange}
          />
        </Card>
      )}
    </Card>
  );
};

export default EmbeddingSearch;
