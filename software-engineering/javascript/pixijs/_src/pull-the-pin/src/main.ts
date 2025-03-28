import { Application, Assets, Sprite } from "pixi.js";
// Import SVGs as URLs (Vite will likely inline them as data URLs)
import pinUrl from "./assets/pin.svg";
import treasureUrl from "./assets/treasure.svg";
import lavaUrl from "./assets/lava.svg";
import heroUrl from "./assets/hero.svg";

(async () => {
  // Create a new application
  const app = new Application();

  // Initialize the application
  await app.init({ background: "#1099bb", resizeTo: window });

  // Append the application canvas to the document body
  document.getElementById("pixi-container")!.appendChild(app.canvas);

  // Add assets using imported URLs
  Assets.add({ alias: "pin", src: pinUrl });
  Assets.add({ alias: "treasure", src: treasureUrl });
  Assets.add({ alias: "lava", src: lavaUrl });
  Assets.add({ alias: "hero", src: heroUrl });

  // Load assets
  const textures = await Assets.load(["pin", "treasure", "lava", "hero"]);

  // --- Pin Objects ---
  // Pin 1 (Top, supports treasure)
  const pin1 = new Sprite(textures.pin);
  pin1.anchor.set(0.5);
  pin1.rotation = Math.PI / 6; // Approx 30 degrees clockwise
  pin1.position.set(app.screen.width * 0.5, app.screen.height * 0.35);
  app.stage.addChild(pin1);

  // Pin 2 (Middle Left, supports lava)
  const pin2 = new Sprite(textures.pin);
  pin2.anchor.set(0.5);
  pin2.rotation = Math.PI / 4; // 45 degrees clockwise
  pin2.position.set(app.screen.width * 0.4, app.screen.height * 0.55);
  app.stage.addChild(pin2);

  // Pin 3 (Middle Right, supports lava)
  const pin3 = new Sprite(textures.pin);
  pin3.anchor.set(0.5);
  pin3.rotation = -Math.PI / 4; // 45 degrees counter-clockwise
  pin3.position.set(app.screen.width * 0.6, app.screen.height * 0.55);
  app.stage.addChild(pin3);

  // --- Treasure Object ---
  const treasure = new Sprite(textures.treasure);
  treasure.anchor.set(0.5);
  // Position treasure above pin1, slightly right
  treasure.position.set(app.screen.width * 0.55, app.screen.height * 0.25);
  app.stage.addChild(treasure);

  // --- Lava Object ---
  // TODO: This is a placeholder visual. Physics/interaction logic will likely
  // change significantly when a physics engine is integrated.
  const lava = new Sprite(textures.lava);
  lava.anchor.set(0.5);
  // Position lava above the crossed pins (pin2, pin3)
  lava.position.set(app.screen.width / 2, app.screen.height * 0.5);
  app.stage.addChild(lava);

  // --- Hero Object ---
  const hero = new Sprite(textures.hero);
  hero.anchor.set(0.5);
  // Position hero at the bottom center
  hero.position.set(app.screen.width / 2, app.screen.height * 0.85);
  app.stage.addChild(hero);

  // No animation needed for now
  // app.ticker.add((time) => {
  // });
})();
