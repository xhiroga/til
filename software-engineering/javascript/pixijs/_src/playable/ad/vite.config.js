import { defineConfig } from 'vite';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Default build configuration
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    }
  },
  // Additional build configuration for Playable Ad
  Playable: {
    build: {
      outDir: 'dist-ad',
      sourcemap: false,
      minify: true,
      cssCodeSplit: false,
      assetsInlineLimit: 100000000, // Inline all assets
      rollupOptions: {
        output: {
          manualChunks: undefined, // Disable code splitting
          inlineDynamicImports: true, // Inline dynamic imports
          entryFileNames: 'assets/[name].js',
          chunkFileNames: 'assets/[name].js',
          assetFileNames: 'assets/[name].[ext]'
        },
        input: {
          main: resolve(__dirname, 'index.html'),
        },
      }
    }
  }
});