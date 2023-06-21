"use client";

import { ApolloProvider } from "@apollo/client";
import { client } from "../../apollo/client";

const WithApollo = ({ children }: React.PropsWithChildren) => {
  return <ApolloProvider client={client}>{children}</ApolloProvider>;
};

export default WithApollo;
