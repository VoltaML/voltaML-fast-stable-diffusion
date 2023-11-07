import { trackRouter } from "vue-gtag-next";
import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: () => import("../views/TextToImageView.vue"),
    },
    {
      path: "/txt2img",
      name: "txt2img",
      component: () => import("../views/TextToImageView.vue"),
    },
    {
      path: "/img2img",
      name: "img2img",
      component: () => import("../views/Image2ImageView.vue"),
    },
    {
      path: "/imageProcessing",
      name: "imageProcessing",
      component: () => import("../views/ImageProcessingView.vue"),
    },
    {
      path: "/models",
      name: "models",
      component: () => import("../views/ModelsView.vue"),
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
      path: "/extra",
      name: "extra",
      component: () => import("../views/ExtraView.vue"),
    },
    {
      path: "/test",
      name: "test",
      component: () => import("../views/TestView.vue"),
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
    {
      path: "/:pathMatch(.*)",
      name: "notFound",
      component: () => import("../views/404View.vue"),
    },
  ],
});

trackRouter(router);

export default router;
