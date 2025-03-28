import { Application, Assets, Sprite } from "pixi.js";

(async () => {
  // Create a new application
  const app = new Application();

  // Initialize the application
  await app.init({ background: "#1099bb", resizeTo: window });

  // Append the application canvas to the document body
  document.getElementById("pixi-container")!.appendChild(app.canvas);

  // Load assets
  Assets.add({ alias: "pin", src: "/assets/pin.svg" });
  Assets.add({ alias: "treasure", src: "/assets/treasure.svg" });
  Assets.add({ alias: "lava", src: "/assets/lava.svg" });
  Assets.add({ alias: "hero", src: "/assets/hero.svg" });
  const textures = await Assets.load(["pin", "treasure", "lava", "hero"]);

  // --- Pin Object ---
  const pin = new Sprite(textures.pin);
  pin.anchor.set(0.5);
  pin.position.set(
    app.screen.width / 2,
    app.screen.height / 2 - pin.height / 2 - 100 // Adjusted position slightly higher
  );
  app.stage.addChild(pin);

  // --- Treasure Object ---
  const treasure = new Sprite(textures.treasure);
  treasure.anchor.set(0.5);
  treasure.position.set(
    app.screen.width / 2,
    app.screen.height / 2 + treasure.height / 2 // Position below center
  );
  app.stage.addChild(treasure);

  // --- Lava Object ---
  // TODO: This is a placeholder visual. Physics/interaction logic will likely
  // change significantly when a physics engine is integrated.
  const lava = new Sprite(textures.lava);
  lava.anchor.set(0.5);
  lava.position.set(
    app.screen.width / 2 - lava.width - 20, // Position left of treasure
    app.screen.height / 2 + lava.height / 2
  );
  app.stage.addChild(lava);

  // --- Hero Object ---
  const hero = new Sprite(textures.hero);
  hero.anchor.set(0.5);
  hero.position.set(
    app.screen.width / 2 + hero.width + 20, // Position right of treasure
    app.screen.height / 2 + hero.height / 2
  );
  app.stage.addChild(hero);

  // No animation needed for now
  // app.ticker.add((time) => {
  // });
})();
