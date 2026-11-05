import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Static assets are emitted straight into the repository-level `public/`
// directory, which the C++ StaticFileHandler serves at runtime.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../public',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
