// https://socket.io/get-started/chat/

const app = require("express")();
const http = require("http").createServer(app);
const io = require("socket.io")(http);

app.get("/", (req, res) => {
  console.log("serve html");
  res.sendFile(__dirname + "/index.html");
});

io.on("connection", (socket) => {
  console.log("a user connected");
  socket.on("chat message", (msg) => {
    // socket.broadcast.emit('hi'); // excluding sender
    io.emit("chat message", msg);
    console.log("message: " + msg);
  });
  socket.on("disconnect", () => {
    console.log("user disconnected.");
  });
});

http.listen(3000, () => {
  console.log("listening on *:3000");
});
