import * as express from "express";

const app = express();
const PORT = 8000;

const namespaces = [];

const v1 = express.Router();
app.use("v1", v1);
v1.route("nodes").get((req, res) => res.send("Express + TypeScript Server"));

app.listen(PORT, () => {
  console.log(`⚡️[server]: Server is running at http://localhost:${PORT}`);
});
