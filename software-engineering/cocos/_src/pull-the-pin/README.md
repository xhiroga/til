# Pull the Pin (with SD)

## How to use

```console
cd mcp-server
pnpm run build
# Type "game-makerで、ハンバーガーとバイ菌でPull-the-Pinゲームを作って"
```

## How to debug

```console
# Terminal #1
cd mcp-server
pnpm run watch

# Terminal #2
wsl
pnpm run inspector
# Command: node, Arguments: /mnt/c/Users/hiroga/GitHub/til/software-engineering/cocos/_src/pull-the-pin/mcp-server/build/index.js
```
