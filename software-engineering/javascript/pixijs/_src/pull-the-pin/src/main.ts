import { Application, Assets, Sprite } from "pixi.js";

(async () => {
  // Create a new application
  const app = new Application();

  // Initialize the application
  await app.init({ background: "#1099bb", resizeTo: window });

  // Append the application canvas to the document body
  document.getElementById("pixi-container")!.appendChild(app.canvas);

  // --- Pin Object ---
  // Load the pin texture from SVG
  const pinTexture = await Assets.load("/assets/pin.svg");

  // Create a pin Sprite
  const pin = new Sprite(pinTexture);

  // Center the sprite's anchor point (optional, but often useful)
  pin.anchor.set(0.5);

  // Position the pin
  // Center horizontally, place it slightly above the vertical center
  pin.position.set(
    app.screen.width / 2,
    app.screen.height / 2 - pin.height / 2 - 50
  );

  // Add the pin to the stage
  app.stage.addChild(pin);

  // No animation needed for now
  // app.ticker.add((time) => {
  // });
})();
