// src/apolloClient.ts
import { ApolloClient, InMemoryCache, HttpLink } from "@apollo/client";
import backendHost from "./env";

const httpLink = new HttpLink({
  uri: `http://${backendHost}:8080/v1/graphql`,
});

const client = new ApolloClient({
  link: httpLink,
  cache: new InMemoryCache(),
});

export default client;
