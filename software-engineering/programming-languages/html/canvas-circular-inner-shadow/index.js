const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const x = 100
const y = 100

context.save();
context.beginPath();
context.lineWidth = 6;
context.shadowColor = 'black';
context.strokeStyle = "rgba(0,0,0,1)";
context.shadowBlur = 15;
context.shadowOffsetX = 10;
context.shadowOffsetY = 10;
context.arc(x, y, 47, 0, 2 * Math.PI, false);
context.stroke();
context.restore();

context.save();
context.beginPath();
context.lineWidth = 6;
context.arc(x, y, 50, 0, 2 * Math.PI);
context.stroke();
context.restore();