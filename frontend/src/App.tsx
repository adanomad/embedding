import React from "react";
import { Container, Paper } from "@mantine/core";
import FileEmbeddingCount from "./FileEmbeddingCount";
import EmbeddingSearch from "./EmbeddingSearch";

const App: React.FC = () => {
  return (
    <Container size="md" className="App">
      <Paper>
        <FileEmbeddingCount />
        <EmbeddingSearch />
      </Paper>
    </Container>
  );
};

export default App;
