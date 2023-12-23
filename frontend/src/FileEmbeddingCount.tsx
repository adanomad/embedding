// src/FileEmbeddingCount.tsx
import React from "react";
import { useQuery, gql } from "@apollo/client";
import { Container, Title, Text, Loader, Card, Space } from "@mantine/core";

const FILE_EMBEDDING_COUNT_QUERY = gql`
  {
    Aggregate {
      FileEmbedding {
        meta {
          count
        }
      }
    }
  }
`;

const FileEmbeddingCount: React.FC = () => {
  const { data, loading, error } = useQuery(FILE_EMBEDDING_COUNT_QUERY, {
    pollInterval: 5000, // Poll every 5000 milliseconds (5 seconds)
  });

  if (loading) return <Loader />;
  if (error) return <Text color="red">Error: {error.message}</Text>;

  const count = data?.Aggregate?.FileEmbedding[0]?.meta?.count;

  return (
    <Container>
      <Card shadow="sm" padding="lg">
        <Title order={2}>File Embedding Count</Title>
        <Space h="md" />
        <Text size="lg">Count: {count}</Text>
      </Card>
    </Container>
  );
};

export default FileEmbeddingCount;
