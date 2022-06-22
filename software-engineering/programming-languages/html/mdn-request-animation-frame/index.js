const element = document.getElementById("some-element-you-want-to-animate");
let start, previousTimeStamp;
let done = false;

const step = (timestamp) => {
  if (start === undefined) {
    start = timestamp;
  }
  const elapsed = timestamp - start;

  if (previousTimeStamp !== timestamp) {
    // ここで Math.min() を使用して、要素がちょうど 200px で止まるようにします。
    const count = Math.min(0.1 * elapsed, 200);
    element.style.transform = "translateX(" + count + "px)";
    if (count === 200) done = true;
  }

  if (elapsed < 2000) {
    // Stop the animation after 2 seconds
    previousTimeStamp = timestamp;
    !done && window.requestAnimationFrame(step);
  }
};

window.requestAnimationFrame(step);
