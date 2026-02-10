import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import cesium from 'vite-plugin-cesium';
import path from 'path';

export default defineConfig({
  plugins: [react(), cesium()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  resolve: {
    alias: {
      // Cesium 1.119 imports @zip.js/zip.js/lib/zip-no-worker.js which was
      // removed in newer @zip.js versions. Map to the equivalent file.
      '@zip.js/zip.js/lib/zip-no-worker.js': path.resolve(
        __dirname, 'node_modules/@zip.js/zip.js/lib/zip-core.js'
      ),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
});
