import { defineConfig } from "vite";

// In dev (`npm run dev`, http://localhost:5173), API calls go to the Python
// server on :7860. In production, set VITE_API_BASE_URL to the server's URL.
export default defineConfig({
  server: {
    proxy: { "/api": "http://localhost:7860" },
  },
});
