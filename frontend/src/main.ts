import { createPinia } from "pinia";
import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";

import "./assets/main.css";

export const pinia = createPinia();
const app = createApp(App);
app.use(pinia);

app.use(router);

app.mount("#app");
