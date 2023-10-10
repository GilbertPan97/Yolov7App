import { createRouter, createWebHashHistory, RouteRecordRaw } from "vue-router";
import About from "../views/About.vue";
import PlatForm from "../views/PlatForm.vue";
// import Analysis from "../views/Analysis.vue";

export const routes: Array<RouteRecordRaw> = [
  {
    path: "/",
    redirect: "/platForm",
    children: [
      {
        path: "/dashboard",
        redirect: "/dashboard/home",
        meta: {
          title: "仪表盘",
          icon: "",
        },
      },
      {
        path: "/platForm",
        component: PlatForm,
        meta: {
          title: "工作台",
          icon: "",
        },
      },
      {
        path: "/about",
        component: About,
        meta: {
          title: "关于",
          icon: "",
        },
      },
    ],
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
