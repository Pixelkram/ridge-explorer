import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,           // listen on 0.0.0.0 for LAN/VPN access
    strictPort: true,
    proxy: {
      '/api': 'http://localhost:8001',
      '/cache': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
    }
  }
})
