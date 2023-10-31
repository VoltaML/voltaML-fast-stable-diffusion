import { createPinia } from "pinia";
import { createApp } from "vue";
import VueGtag from "vue-gtag-next";
import App from "./App.vue";
import router from "./router";

import "./assets/main.css";

const app = createApp(App);

app.use(createPinia());
app.use(router);
app.use(VueGtag, {
  property: {
    id: "G-PYLCYXF7B8",
  },
});

app.mount("#app");
