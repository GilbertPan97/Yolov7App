<template>
  <a-layout style="min-height: 100vh">
    <a-layout-sider v-model:collapsed="collapsed" :trigger="null" collapsible>
      <a-space :size="20" class="logo">
        <img src="../../public/logo.svg" class="logoImg" />
        <template v-if="!collapsed">
          <span style="font-weight: bold; font-size: 25px">ROBOT</span>
        </template>
      </a-space>
      <a-menu
        v-model:openKeys="openKeys"
        v-model:selectedKeys="selectedKeys"
        theme="dark"
        mode="inline"
      >
        <router-menu-item
          v-for="(route, index) in routes"
          :key="index"
          :route="route"
      /></a-menu>
    </a-layout-sider>
    <!-- 自定义头部和底部 -->
    <a-layout>

      <a-layout-content class="content">
        <router-view></router-view>
      </a-layout-content>

      <a-layout-footer class="footer">
        <span>YOLO APP ©2023 Created by Jiabin Pan</span>
      </a-layout-footer>
    </a-layout>
  </a-layout>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import router from "../router/router";
import RouterMenuItem from "../components/RouterMenuItem.vue";

const routes = router.options.routes[0].children;
const collapsed = ref<boolean>(false);
const selectedKeys = ref<string[]>(["/platForm"]);
const openKeys = ref<string[]>(["/dashboard"]);
</script>

<style scoped>
.logo {
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 50px;
  margin: 15px;
  background: rgba(255, 255, 255, 0.6);
}

.logoImg {
  height: 38px;
}

.site-layout .site-layout-background {
  background: #fff;
}
[data-theme="dark"] .site-layout .site-layout-background {
  background: #141414;
}

.trigger {
  font-size: 20px;
}
.header {
  background: #fff;
  height: 50px;
  padding: 0 15px;
  display: flex;
  align-items: center;
}
.content {
  margin: 10px 10px 0 10px;
  background: #fff;
  min-height: 82%;
}
.footer {
  align-items: center;
  justify-content: center;
  display: flex;
  max-height: 20px;
  padding: 0;
  margin: 5px 0;
}
</style>
