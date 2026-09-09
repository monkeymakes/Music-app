import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// SPA-only: @netlify/vite-plugin is omitted in local dev because its edge-functions
// emulator can fail in constrained environments. netlify.toml still drives deploy.
export default defineConfig({
  plugins: [react()],
})
