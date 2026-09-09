import react from '@vitejs/plugin-react'
import netlify from '@netlify/vite-plugin'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), netlify()],
})
