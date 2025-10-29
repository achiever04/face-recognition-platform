// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  // --- ADD THIS CONTENT SECTION ---
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // --- END ADD ---
  theme: {
    extend: {},
  },
  plugins: [],
}