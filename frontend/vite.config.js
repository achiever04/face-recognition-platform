// This file configures Vite, the frontend build tool.
// - It imports `defineConfig` from Vite for configuration.
// - It imports the Tailwind CSS Vite plugin (`@tailwindcss/vite`).
// - It imports the React Vite plugin (`@vitejs/plugin-react`).
// - It exports the configuration object, enabling both Tailwind and React plugins.
// This is the standard way to set up Vite for a React + Tailwind project.
// No improvements needed.

import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite' // Ensure you have installed `@tailwindcss/vite`
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(), // Enables Tailwind CSS processing
    react()        // Enables React support (JSX, Fast Refresh, etc.)
  ],
  // Add other Vite configurations here if needed (e.g., server proxy, base path)
})