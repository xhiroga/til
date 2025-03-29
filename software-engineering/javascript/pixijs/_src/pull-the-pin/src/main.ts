import { Application, Assets, Sprite, Container } from "pixi.js";
import * as Matter from "matter-js"; // Import Matter.js

// Import SVGs as URLs
import pinUrl from "./assets/pin.svg";
import treasureUrl from "./assets/treasure.svg";
import lavaUrl from "./assets/lava.svg";
import heroUrl from "./assets/hero.svg";
import backgroundUrl from "./assets/background.svg";
import wallUrl from "./assets/wall.svg";
import slopeUrl from "./assets/slope.svg";

(async () => {
  // Create a new application
  const app = new Application();

  // Initialize the application
  await app.init({ background: "#1099bb", resizeTo: window });

  // Append the application canvas to the document body
  document.getElementById("pixi-container")!.appendChild(app.canvas);

  // --- Matter.js Setup ---
  const engine = Matter.Engine.create();
  const world = engine.world;
  // Disable gravity initially or set as needed
  // engine.world.gravity.y = 0; // Example: No gravity initially

  // Add assets using imported URLs
  Assets.add({ alias: "pin", src: pinUrl });
  Assets.add({ alias: "treasure", src: treasureUrl });
  Assets.add({ alias: "lava", src: lavaUrl });
  Assets.add({ alias: "hero", src: heroUrl });
  Assets.add({ alias: "background", src: backgroundUrl });
  Assets.add({ alias: "wall", src: wallUrl });
  Assets.add({ alias: "slope", src: slopeUrl });

  // Load assets
  const textures = await Assets.load(["pin", "treasure", "lava", "hero", "background", "wall", "slope"]);

  // Create game container (for organizing layers)
  const gameContainer = new Container();
  app.stage.addChild(gameContainer);

  // --- Background Layer ---
  const background = new Sprite(textures.background);
  background.width = app.screen.width;
  background.height = app.screen.height;
  gameContainer.addChild(background);

  // --- Wall Layer (Visual) ---
  // Keep the visual wall sprite
  const wallSprite = new Sprite(textures.wall);
  wallSprite.anchor.set(0.5, 0);
  wallSprite.position.set(app.screen.width * 0.5, 0);
  wallSprite.width = app.screen.width * 0.7;
  wallSprite.height = app.screen.height;
  gameContainer.addChild(wallSprite);

  // --- Matter.js Walls and Ground ---
  const wallThickness = 50; // Thickness for physics boundaries
  const wallOptions = { isStatic: true, render: { visible: false } }; // Invisible static bodies

  // Ground
  const ground = Matter.Bodies.rectangle(
    app.screen.width / 2,
    app.screen.height + wallThickness / 2, // Position below screen
    app.screen.width,
    wallThickness,
    wallOptions
  );

  // Left Wall (aligned with the visual wall edge)
  const leftWall = Matter.Bodies.rectangle(
    app.screen.width * 0.15 - wallThickness / 2, // Position at the left edge of the visual wall
    app.screen.height / 2,
    wallThickness,
    app.screen.height,
    wallOptions
  );

  // Right Wall (aligned with the visual wall edge)
  const rightWall = Matter.Bodies.rectangle(
    app.screen.width * 0.85 + wallThickness / 2, // Position at the right edge of the visual wall
    app.screen.height / 2,
    wallThickness,
    app.screen.height,
    wallOptions
  );

  Matter.Composite.add(world, [ground, leftWall, rightWall]);


  // --- Slope Object (Visual and Physics) ---
  const slopeSprite = new Sprite(textures.slope);
  slopeSprite.anchor.set(0, 0); // Set anchor to top-left
  const slopePosX = app.screen.width * 0.35;
  const slopePosY = app.screen.height * 0.28;
  const slopeScaleX = 1.6;
  const slopeScaleY = 1.2;
  slopeSprite.position.set(slopePosX, slopePosY);
  slopeSprite.scale.set(slopeScaleX, slopeScaleY);
  gameContainer.addChild(slopeSprite);

  // Create Matter.js body for the slope (approximated as a rectangle for now)
  // Adjust position and size based on the sprite's placement and scale
  const slopeWidth = slopeSprite.width;
  const slopeHeight = slopeSprite.height;
  const slopeBody = Matter.Bodies.rectangle(
      slopePosX + slopeWidth / 2, // Center X
      slopePosY + slopeHeight / 2, // Center Y
      slopeWidth,
      slopeHeight,
      { isStatic: true, angle: 0, label: "slope" } // Static body
  );
  // TODO: Improve slope shape using vertices if needed
  Matter.Composite.add(world, slopeBody);


  // --- Pin Objects (Visual and Physics) ---
  const pins: { sprite: Sprite, body: Matter.Body }[] = [];
  const pinOptions = { isStatic: true, render: { visible: false } }; // Pins are static initially

  // Function to create a pin (Sprite + Body)
  const createPin = (x: number, y: number, rotation: number, scaleX: number, scaleY: number) => {
    const sprite = new Sprite(textures.pin);
    sprite.anchor.set(0.1, 0.5); // Adjust anchor
    sprite.rotation = rotation;
    sprite.position.set(x, y);
    sprite.scale.set(scaleX, scaleY);
    gameContainer.addChild(sprite);

    // Create Matter.js body for the pin (rectangle approximation)
    // Adjust dimensions and position based on sprite scale and rotation
    const pinWidth = sprite.width * 0.8; // Effective width (adjust as needed)
    const pinHeight = sprite.height * 0.5; // Effective height (adjust as needed)
    const body = Matter.Bodies.rectangle(
        x, // Use sprite's x/y as center for the body
        y,
        pinWidth,
        pinHeight,
        { ...pinOptions, angle: rotation, label: `pin-${pins.length + 1}` }
    );
    Matter.Composite.add(world, body);
    pins.push({ sprite, body });

    // Add interactivity (example: click to remove)
    sprite.eventMode = 'static';
    sprite.cursor = 'pointer';
    sprite.on('pointerdown', () => {
        console.log(`Pin clicked: ${body.label}`);
        // Remove from Matter world
        Matter.Composite.remove(world, body);
        // Remove Pixi sprite
        gameContainer.removeChild(sprite);
        // Remove from our pins array (optional, depends on further logic)
        const index = pins.findIndex(p => p.body === body);
        if (index > -1) {
            pins.splice(index, 1);
        }
    });
  };

  // Create the pins using the function
  createPin(app.screen.width * 0.42, app.screen.height * 0.32, Math.PI / 5.5, 3.0, 1.0); // Pin 1
  createPin(app.screen.width * 0.32, app.screen.height * 0.58, Math.PI / 3.2, 3.2, 1.0); // Pin 2
  createPin(app.screen.width * 0.68, app.screen.height * 0.58, -Math.PI / 3.2, 3.2, 1.0); // Pin 3


  // --- Treasure Objects (Visual and Physics) ---
  const treasureSprites: Sprite[] = [];
  const treasureBodies: Matter.Body[] = [];
  const treasureGroupContainer = new Container(); // Use a container for visual grouping if needed
  // treasureGroupContainer.position.set(app.screen.width * 0.53, app.screen.height * 0.35); // Initial group position - Let bodies dictate position
  gameContainer.addChild(treasureGroupContainer); // Add container, but sprites will be positioned absolutely in the world

  const createTreasure = (initialX: number, initialY: number, scale: number) => {
      const sprite = new Sprite(textures.treasure);
      sprite.anchor.set(0.5);
      sprite.scale.set(scale);
      // Initial sprite position matches body position
      sprite.position.set(initialX, initialY);
      // Add sprite directly to the main game container, not the group container
      gameContainer.addChild(sprite);
      treasureSprites.push(sprite);

      const body = Matter.Bodies.circle(
          initialX,
          initialY,
          sprite.width / 2, // Radius based on scaled sprite width
          { restitution: 0.3, friction: 0.5, label: "treasure" } // Add some bounce and friction
      );
      treasureBodies.push(body);
      Matter.Composite.add(world, body);
  };

  // Calculate initial absolute positions for treasure based on original group logic
  const groupBaseX = app.screen.width * 0.53;
  const groupBaseY = app.screen.height * 0.35;

  createTreasure(groupBaseX + 0, groupBaseY + 0, 0.45);    // Main treasure
  createTreasure(groupBaseX - 15, groupBaseY + 3, 0.4);   // Treasure 2
  createTreasure(groupBaseX + 15, groupBaseY + 3, 0.4);    // Treasure 3
  createTreasure(groupBaseX + 0, groupBaseY - 10, 0.35);  // Treasure 4
  createTreasure(groupBaseX - 10, groupBaseY - 5, 0.38); // Treasure 5


  // --- Lava Object (Visual and Physics) ---
  // Visual representation (main blob)
  const lavaSprite = new Sprite(textures.lava);
  lavaSprite.anchor.set(0.5);
  const lavaInitialX = app.screen.width / 2;
  const lavaInitialY = app.screen.height * 0.54;
  lavaSprite.position.set(lavaInitialX, lavaInitialY);
  lavaSprite.scale.set(1.8);
  gameContainer.addChild(lavaSprite);

  // Physics Body for Lava (approximated as a circle)
  const lavaBody = Matter.Bodies.circle(
      lavaInitialX,
      lavaInitialY,
      lavaSprite.width / 2 * 0.8, // Approximate radius, adjust as needed
      { restitution: 0.1, friction: 0.8, density: 0.005, label: "lava" } // Lava properties
  );
  Matter.Composite.add(world, lavaBody);

  // Remove old lava particles and effects if they are purely visual
  // gameContainer.removeChild(lavaParticle1, lavaParticle2); // Assuming these were purely visual


  // --- Hero Object (Visual Only for now) ---
  const hero = new Sprite(textures.hero);
  hero.anchor.set(0.5);
  hero.position.set(app.screen.width * 0.35, app.screen.height * 0.89);
  hero.scale.set(1.15);
  gameContainer.addChild(hero);

  // Remove other decorative elements if they don't interact physically
  // gameContainer.removeChild(candle1, candle2, skull);


  // --- Game Loop Integration ---
  app.ticker.add((ticker) => {
    // Matter.js recommends using a fixed delta time for stability,
    // but using ticker.deltaMS is also common. Let's try ticker's delta.
    // Use deltaMS as Matter.Engine.update expects milliseconds
    const delta = ticker.deltaMS;

    // Update Matter.js engine
    Matter.Engine.update(engine, delta); // Update engine

    // Update Lava Sprite position and rotation from Matter.js body
    lavaSprite.position.set(lavaBody.position.x, lavaBody.position.y);
    lavaSprite.rotation = lavaBody.angle;
    // Keep the visual pulsing effect if desired
    lavaSprite.scale.x = 1.8 + Math.sin(app.ticker.lastTime / 300) * 0.1;
    lavaSprite.scale.y = 1.8 + Math.sin(app.ticker.lastTime / 300) * 0.1;


    // Update Treasure Sprites positions and rotations from Matter.js bodies
    for (let i = 0; i < treasureSprites.length; i++) {
        treasureSprites[i].position.set(treasureBodies[i].position.x, treasureBodies[i].position.y);
        treasureSprites[i].rotation = treasureBodies[i].angle;
    }


    // Update remaining Pin Sprites (if any left) - they are static unless removed
    // No update needed for static pins' sprites unless they become dynamic


    // Keep other visual animations if needed
    // hero.y = app.screen.height * 0.89 + Math.sin(app.ticker.lastTime / 500) * 2;

  });

  // Optional: Add Matter.js Renderer for debugging
  /*
  const render = Matter.Render.create({
      element: document.body, // Render to the body or a specific element
      engine: engine,
      options: {
          width: app.screen.width,
          height: app.screen.height,
          wireframes: true, // Show wireframes
          showAngleIndicator: true
      }
  });
  Matter.Render.run(render);
  */

})();
