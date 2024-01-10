import { createRouter, createWebHashHistory, RouteRecordRaw } from "vue-router";
import Detectron from "../views/Detectron.vue";
import Segmentation from "../views/Segmentation.vue";
import Estimation from "../views/Estimation.vue";

export const routes: Array<RouteRecordRaw> = [
  {
    path: "/",
    redirect: "/platForm",
    children: [
      {
        path: "/dashboard",
        redirect: "/dashboard/home",
        meta: {
          title: "Dashboard",
          icon: "",
        },
      },
      {
        path: "/detectron",
        component: Detectron,
        meta: {
          title: "Detectron",
          icon: "",
        },
      },
      {
        path: "/segmentation",
        component: Segmentation,
        meta: {
          title: "Segmentation",
          icon: "",
        },
      },
      {
        path: "/estimation",
        component: Estimation,
        meta: {
          title: "Estimation",
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
