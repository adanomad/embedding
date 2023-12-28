import weaviate from "weaviate-ts-client";

export const client = weaviate.client({
  scheme: "http",
  host: "glassbox.ds:8080",
});
