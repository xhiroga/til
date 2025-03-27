import { defineConfig } from 'vite';
import { resolve } from 'path';
import fs from 'fs';

// Configuration specifically for Playable Ad ad builds
export default defineConfig({
  build: {
    outDir: 'dist-ad',
    sourcemap: false,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    // Inline all assets as base64
    assetsInlineLimit: 100000000,
    // Don't split CSS into separate files
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        // Create a single JS file
        format: 'iife',
        // Disable code splitting
        manualChunks: undefined,
        // Inline dynamic imports
        inlineDynamicImports: true,
        // Ensure all code is in a single file
        compact: true,
        // Use a single entry point
        entryFileNames: 'bundle.js',
      },
      input: resolve(__dirname, 'index.html'),
    }
  },
  plugins: [
    {
      name: 'generate-single-html',
      closeBundle: async () => {
        // Read the generated HTML and JS files
        const htmlPath = resolve(__dirname, 'dist-ad/index.html');
        const jsPath = resolve(__dirname, 'dist-ad/bundle.js');
        
        let htmlContent = fs.readFileSync(htmlPath, 'utf-8');
        const jsContent = fs.readFileSync(jsPath, 'utf-8');
        
        // Replace the script tag with an inline script
        htmlContent = htmlContent.replace(
          /<script[^>]*src=["']\/bundle\.js["'][^>]*><\/script>/,
          `<script>${jsContent}</script>`
        );
        
        // Write the combined HTML file
        fs.writeFileSync(resolve(__dirname, 'dist-ad/Playable Ad-ad.html'), htmlContent);
        
        console.log('Single HTML file generated at dist-ad/Playable Ad-ad.html');
      }
    }
  ]
});