import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // 如果你的请求URL类似 'api/xxx'
      "/api": {
        target: "http://your-backend-server.com", // 这里是你的后端服务地址
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""), // 下面这行可参与根据你的实际情况决定是否需要
      },
    },
  },
});
