import { createApp } from 'vue';
import App from './App.vue';
import 'ant-design-vue/dist/reset.css';
import Antd from 'ant-design-vue'
import router from './router/router';

// 根据你的需求，在这里可以导入其他必要的模块

const app = createApp(App);

// 在这里可以进行其他配置，如添加全局组件、混入等

app.use(router).use(Antd).mount('#app');