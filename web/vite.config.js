import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// host: true so a phone on the same Wi-Fi can reach the dev server by LAN IP.
export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173 },
})
