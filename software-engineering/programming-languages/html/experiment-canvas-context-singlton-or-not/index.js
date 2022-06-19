const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const ctx2 = canvas.getContext("2d");

ctx.beginPath();
ctx.arc(30, 30, 25, 0, Math.PI * 2, true);
ctx.closePath();

ctx.fillStyle = "blue";
ctx2.fillStyle = "red";
ctx.fill(); // is blue or red? → red!!!
