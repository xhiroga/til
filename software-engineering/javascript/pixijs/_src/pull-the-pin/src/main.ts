import { Application, Assets, Sprite, Container } from "pixi.js";
// Import SVGs as URLs (Vite will likely inline them as data URLs)
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

  // --- Wall Layer ---
  const wall = new Sprite(textures.wall);
  wall.anchor.set(0.5, 0);
  wall.position.set(app.screen.width * 0.5, 0);
  wall.width = app.screen.width * 0.7;
  wall.height = app.screen.height;
  gameContainer.addChild(wall);

  // --- Slope Object (for treasure to rest on) ---
  const slope = new Sprite(textures.slope);
  slope.anchor.set(0, 0); // Set anchor to top-left
  slope.position.set(app.screen.width * 0.35, app.screen.height * 0.28);
  slope.scale.set(1.6, 1.2);
  gameContainer.addChild(slope);

  // --- Pin Objects ---
  // Pin 1 (Top, supports treasure)
  const pin1 = new Sprite(textures.pin);
  pin1.anchor.set(0.1, 0.5); // Adjust anchor for the new SVG's handle position
  pin1.rotation = Math.PI / 5.5; // Approx 32 degrees clockwise
  pin1.position.set(app.screen.width * 0.42, app.screen.height * 0.32);
  pin1.scale.set(3.0, 1.0); // Make pin longer by scaling x more than y
  gameContainer.addChild(pin1);

  // Pin 2 (Middle Left, supports lava)
  const pin2 = new Sprite(textures.pin);
  pin2.anchor.set(0.1, 0.5);
  pin2.rotation = Math.PI / 3.2; // About 56 degrees clockwise
  pin2.position.set(app.screen.width * 0.32, app.screen.height * 0.58);
  pin2.scale.set(3.2, 1.0);
  gameContainer.addChild(pin2);

  // Pin 3 (Middle Right, supports lava)
  const pin3 = new Sprite(textures.pin);
  pin3.anchor.set(0.1, 0.5);
  pin3.rotation = -Math.PI / 3.2; // About 56 degrees counter-clockwise
  pin3.position.set(app.screen.width * 0.68, app.screen.height * 0.58);
  pin3.scale.set(3.2, 1.0);
  gameContainer.addChild(pin3);

  // --- Treasure Objects ---
  // コインをグループ化して配置
  const treasureGroup = new Container();
  treasureGroup.position.set(app.screen.width * 0.53, app.screen.height * 0.35);
  gameContainer.addChild(treasureGroup);
  
  // メインのコイン
  const treasure = new Sprite(textures.treasure);
  treasure.anchor.set(0.5);
  treasure.scale.set(0.45);
  treasureGroup.addChild(treasure);
  
  // 追加のコイン（少しずらして重ねる）
  const treasure2 = new Sprite(textures.treasure);
  treasure2.anchor.set(0.5);
  treasure2.position.set(-15, 3);
  treasure2.scale.set(0.4);
  treasureGroup.addChild(treasure2);
  
  const treasure3 = new Sprite(textures.treasure);
  treasure3.anchor.set(0.5);
  treasure3.position.set(15, 3);
  treasure3.scale.set(0.4);
  treasureGroup.addChild(treasure3);
  
  // さらに追加のコイン
  const treasure4 = new Sprite(textures.treasure);
  treasure4.anchor.set(0.5);
  treasure4.position.set(0, -10);
  treasure4.scale.set(0.35);
  treasureGroup.addChild(treasure4);
  
  const treasure5 = new Sprite(textures.treasure);
  treasure5.anchor.set(0.5);
  treasure5.position.set(-10, -5);
  treasure5.scale.set(0.38);
  treasureGroup.addChild(treasure5);

  // --- Lava Object ---
  // TODO: This is a placeholder visual. Physics/interaction logic will likely
  // change significantly when a physics engine is integrated.
  const lava = new Sprite(textures.lava);
  lava.anchor.set(0.5);
  // Position lava above the crossed pins (pin2, pin3)
  lava.position.set(app.screen.width / 2, app.screen.height * 0.54);
  lava.scale.set(1.8);
  gameContainer.addChild(lava);

  // 火の粒子効果（小さな溶岩の粒）
  const lavaParticle1 = new Sprite(textures.lava);
  lavaParticle1.anchor.set(0.5);
  lavaParticle1.position.set(app.screen.width * 0.48, app.screen.height * 0.5);
  lavaParticle1.scale.set(0.2);
  lavaParticle1.alpha = 0.6;
  gameContainer.addChild(lavaParticle1);
  
  const lavaParticle2 = new Sprite(textures.lava);
  lavaParticle2.anchor.set(0.5);
  lavaParticle2.position.set(app.screen.width * 0.52, app.screen.height * 0.49);
  lavaParticle2.scale.set(0.15);
  lavaParticle2.alpha = 0.5;
  gameContainer.addChild(lavaParticle2);

  // --- Hero Object ---
  const hero = new Sprite(textures.hero);
  hero.anchor.set(0.5);
  // Position hero at the bottom left of center
  hero.position.set(app.screen.width * 0.35, app.screen.height * 0.89);
  hero.scale.set(1.15);
  gameContainer.addChild(hero);
  
  // ヒーローの近くに小道具（キャンドルなど）を追加
  const candle1 = new Sprite(textures.lava);
  candle1.anchor.set(0.5);
  candle1.position.set(app.screen.width * 0.32, app.screen.height * 0.88);
  candle1.scale.set(0.2);
  candle1.alpha = 0.8;
  gameContainer.addChild(candle1);
  
  const candle2 = new Sprite(textures.lava);
  candle2.anchor.set(0.5);
  candle2.position.set(app.screen.width * 0.28, app.screen.height * 0.88);
  candle2.scale.set(0.15);
  candle2.alpha = 0.7;
  gameContainer.addChild(candle2);
  
  // 骸骨のような装飾（簡易的な表現）
  const skull = new Sprite(textures.hero);
  skull.anchor.set(0.5);
  skull.position.set(app.screen.width * 0.68, app.screen.height * 0.88);
  skull.scale.set(0.5);
  skull.tint = 0xFFFFFF; // 白色に
  gameContainer.addChild(skull);

  // Simple animation effect
  app.ticker.add(() => {
    // Subtle pulsing effect for lava
    lava.scale.x = 1.8 + Math.sin(app.ticker.lastTime / 300) * 0.1;
    lava.scale.y = 1.8 + Math.sin(app.ticker.lastTime / 300) * 0.1;
    
    // 溶岩粒子のアニメーション
    lavaParticle1.y = app.screen.height * 0.5 + Math.sin(app.ticker.lastTime / 200) * 5;
    lavaParticle1.alpha = 0.6 + Math.sin(app.ticker.lastTime / 250) * 0.2;
    
    lavaParticle2.y = app.screen.height * 0.49 + Math.cos(app.ticker.lastTime / 150) * 3;
    lavaParticle2.alpha = 0.5 + Math.cos(app.ticker.lastTime / 200) * 0.3;
    
    // キャンドルのゆらぎ効果
    candle1.alpha = 0.7 + Math.sin(app.ticker.lastTime / 200) * 0.2;
    candle2.alpha = 0.6 + Math.cos(app.ticker.lastTime / 150) * 0.2;
    
    // コインの輝き効果
    treasureGroup.rotation = Math.sin(app.ticker.lastTime / 2000) * 0.02;
    
    // 勇者のわずかな動き
    hero.y = app.screen.height * 0.89 + Math.sin(app.ticker.lastTime / 500) * 2;
  });
})();
