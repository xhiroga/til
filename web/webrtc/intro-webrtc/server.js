const express = require("express");
const app = express();
const http = require("http").createServer(app);
const io = require("socket.io")(http);

io.on("connection", (socket) => {
  socket.on("ready", (req) => {
    const room = req;
    socket.join(room);
    socket.broadcast.emit("announce", {
      message: `New Client in the room: ${room}`,
    });
  });
  socket.on("send", (req) => {
    console.log(req);
    socket.join(req.room);
    socket.broadcast.emit("message", {
      message: req.message,
      author: req.author,
    });
  });
});

app.use(express.static(__dirname + "/public"));

app.get("/", (req, res) => {
  res.render("index.ejs");
});

PORT = 3000;
http.listen(PORT, () => {
  console.log("server started on port " + PORT);
});
