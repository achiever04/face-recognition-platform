import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
  ],
  server: {
    // useful during local development to allow external devices to access the dev server
    host: true,
    port: 5173,
  },
});
