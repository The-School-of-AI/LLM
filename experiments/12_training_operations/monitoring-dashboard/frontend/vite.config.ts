import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
        },
      },
    },
  },
  server: {
    proxy: {
      '/stream': 'http://localhost:8000',
      '/runs': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/checkpoints': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
