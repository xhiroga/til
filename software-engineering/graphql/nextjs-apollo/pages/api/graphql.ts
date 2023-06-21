import { readFileSync } from "fs";
import { join } from "path";
import { startServerAndCreateNextHandler } from "@as-integrations/next";
import { ApolloServer } from "@apollo/server";
import { Resolvers } from "../../apollo/__generated__/server/resolvers-types";

const typeDefs = readFileSync(
  join(process.cwd(), "apollo/documents/schema.gql"),
  "utf-8"
);

const resolvers: Resolvers = {
  Query: {
    users() {
      return [{ name: "Nextjs" }, { name: "Nuxtjs" }, { name: "Sveltekit" }];
    },
  },
};

const apolloServer = new ApolloServer<Resolvers>({ typeDefs, resolvers });

export default startServerAndCreateNextHandler(apolloServer);
