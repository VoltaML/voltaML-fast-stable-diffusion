import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "text2image",
      component: () => import("../views/TextToImageView.vue"),
    },
    {
      path: "/image2image",
      name: "image2image",
      component: () => import("../views/Image2ImageView.vue"),
    },
    {
      path: "/extra",
      name: "extra",
      component: () => import("../views/ExtraView.vue"),
    },
    {
      path: "/download",
      name: "download",
      component: () => import("../views/DownloadView.vue"),
    },
    {
      path: "/about",
      name: "about",
      component: () => import("../views/AboutView.vue"),
    },
    {
      path: "/accelerate",
      name: "accelerate",
      component: () => import("../views/AccelerateView.vue"),
    },
    {
      path: "/test",
      name: "test",
      component: () => import("../views/TestView.vue"),
    },
    {
      path: "/convert",
      name: "convert",
      component: () => import("../views/ConvertView.vue"),
    },
    {
      path: "/settings",
      name: "settings",
      component: () => import("../views/SettingsView.vue"),
    },
    {
      path: "/imageBrowser",
      name: "imageBrowser",
      component: () => import("../views/ImageBrowserView.vue"),
    },
    {
      path: "/tagger",
      name: "tagger",
      component: () => import("../views/TaggerView.vue"),
    },
  ],
});

export default router;
