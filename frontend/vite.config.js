import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy all /api calls to FastAPI backend
      '/query': 'http://localhost:8000',
      '/voice-query': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  }
})
