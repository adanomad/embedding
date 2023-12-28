import React from "react";
import { Container, Paper } from "@mantine/core";
import EmbeddingSearch from "./EmbeddingSearch";

const App: React.FC = () => {
  return (
    <Container size="md" className="App">
      <Paper>
        <EmbeddingSearch />
      </Paper>
    </Container>
  );
};

export default App;
