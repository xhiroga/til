// https://www.js-tutorials.com/nodejs-tutorial/simple-websocket-example-with-nodejs/

const WebSocket = require("ws");
const url = "ws://localhost:8080";
const connection = new WebSocket(url);

connection.onopen = () => {
  connection.send("Message From Client");
};

connection.onerror = (error) => {
  console.log(`WebSocket error: ${error}`);
};

connection.onmessage = (event) => {
  console.log(event.data);
};
