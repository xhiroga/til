const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
var raf;

const radius = 25;

const ball = {
  x: 100,
  y: 100,
  vx: 5,
  vy: 2,
  radius,
  color: "blue",
  draw: function () {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.fillStyle = this.color;
    ctx.fill();
  },
};

const draw = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ball.draw();
  if (
    ball.y + ball.vy + radius > canvas.height ||
    ball.y + ball.vy - radius < 0
  ) {
    ball.vy = -ball.vy;
  } else {
    ball.y += ball.vy;
  }
  if (
    ball.x + ball.vx + radius > canvas.width ||
    ball.x + ball.vx - radius < 0
  ) {
    ball.vx = -ball.vx;
  } else {
    ball.x += ball.vx;
  }
  raf = window.requestAnimationFrame(draw);
};

canvas.addEventListener("mouseover", function (e) {
  raf = window.requestAnimationFrame(draw);
});

canvas.addEventListener("mouseout", function (e) {
  window.cancelAnimationFrame(raf);
});

ball.draw();
