const express = require("express");
const app = express();
const http = require("http").createServer(app);
const io = require("socket.io")(http);

io.on("connection", (socket) => {
  socket.on("ready", (req) => {
    const room = req.chat_room;
    const signal_room = req.signal_room;
    socket.join(room);
    socket.join(signal_room);
    socket.broadcast.emit("announce", {
      message: `New Client in the room: ${room}`,
    });
  });
  socket.on("send", (req) => {
    console.log(req);
    socket.join(req.room);
    // if use socket.emit(), message is emitted to all client including sender.
    socket.broadcast.emit("message", {
      message: req.message,
      author: req.author,
    });
  });
  socket.on("signal", (req) => {
    socket.broadcast.emit("signaling_message", {
      type: req.type,
      message: req.message,
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
