import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// El frontend habla con el backend FastAPI. Para evitar problemas de CORS en
// desarrollo, redirigimos /api al servidor de Python (puerto 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  // 'vite preview' sirve el build de producción; replicamos el proxy para que
  // las llamadas a /api lleguen al backend también en este modo.
  preview: {
    port: 4174,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
