<template>
  <div>
    <a-menu-item
      v-if="!route.children || route.children.length == 0"
      :key="route.path"
    >
      <router-link :to="route.path">
        <a-space :size="10">
          <fund-filled style="font-size: 17px" />
          {{ route.meta.title }}
        </a-space>
      </router-link>
    </a-menu-item>
    <a-sub-menu
      v-if="!(!route.children || route.children.length == 0)"
      :key="route.path"
    >
      <template #title>
        <a-space :size="10">
          <fund-filled style="font-size: 17px" />
          {{ route.meta.title }}
        </a-space>
      </template>
      <router-menu-item
        v-for="(childRoute, index) in route.children"
        :key="index"
        :route="childRoute"
      />
    </a-sub-menu>
  </div>
</template>

<script lang="ts">
import {
  FundFilled,
  HomeOutlined,
  MessageFilled,
} from "@ant-design/icons-vue/lib/icons";
import { defineComponent } from "vue";

export default defineComponent({
  components: {
    FundFilled,
    HomeOutlined,
    MessageFilled,
  },
  name: "RouterMenuItem",
  props: {
    route: {
      type: Object,
      required: true,
    },
  },
});
</script>

<style scoped>
.menu {
  font-size: 17px;
  display: flex;
  align-items: center;
}
</style>
