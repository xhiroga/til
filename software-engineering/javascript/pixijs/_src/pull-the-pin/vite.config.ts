import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import svgo from "vite-plugin-svgo";

// https://vite.dev/config/
export default defineConfig({
  plugins: [viteSingleFile(), svgo()],
  build: {
    assetsInlineLimit: 0, // Inline all assets
  },
  server: {
    port: 8080,
    open: true,
  },
});
