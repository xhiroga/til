import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// https://vite.dev/config/
export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    assetsInlineLimit: 0, // Inline all assets
  },
  server: {
    port: 8080,
    open: true,
  },
});
