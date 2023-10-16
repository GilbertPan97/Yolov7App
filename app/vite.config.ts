import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from "path"
import px2rem from "postcss-px2rem"
export default defineConfig({
  base: "./",
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': path.resolve('./src') // @代替src
    }
  },
  server: {
    open: false,//启动项目自动弹出浏览器
    port: 4000,//启动端口
    proxy: {
      '/api': {
        // target: 'http://192.168.20.22:8100',	//实际请求地址
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/api/, '')
      },
    }
  },
  css: {
    preprocessorOptions: {
      less: {
        modifyVars: {
          hack: `true; @import (reference) "${path.resolve("src/assets/css/home.less")}";`,
        },
        javascriptEnabled: true
      }
    },
    // postcss: {
    //   plugins: [
    //     px2rem({
    //       remUnit: 192
    //     })
    //   ]
    // }
  }
})
