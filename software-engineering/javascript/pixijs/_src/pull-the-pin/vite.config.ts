import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// https://vite.dev/config/
export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    // outDir: "singlefile", // Output to default 'dist'
    assetsInlineLimit: 0, // Inline all assets
  },
  server: {
    port: 8080,
    open: true,
  },
});
