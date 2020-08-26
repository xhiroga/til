const express = require("express");
const app = express();

console.log("server started");

app.get("/", (req, res) => {
  res.render("index.ejs");
});

app.listen(3000);
