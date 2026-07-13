import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/offer': 'http://localhost:8000',
      '/state': 'http://localhost:8000',
      '/mode': 'http://localhost:8000',
      '/capture': 'http://localhost:8000',
      '/train': 'http://localhost:8000',
      '/dataset': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
