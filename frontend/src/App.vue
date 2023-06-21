<template>
  <NConfigProvider :theme="theme" :theme-overrides="overrides" class="main">
    <NThemeEditor v-if="settings.data.settings.frontend.enable_theme_editor" />
    <NNotificationProvider placement="bottom-right">
      <NMessageProvider>
        <CollapsileNavbarVue />
        <TopBarVue />
        <routerContainerVue style="margin-top: 52px" />
        <PerformanceDrawer />
      </NMessageProvider>
    </NNotificationProvider>
  </NConfigProvider>
</template>

<script setup lang="ts">
import {
  NConfigProvider,
  NMessageProvider,
  NNotificationProvider,
  NThemeEditor,
  darkTheme,
  lightTheme,
  type GlobalThemeOverrides,
} from "naive-ui";
import { computed } from "vue";
import CollapsileNavbarVue from "./components/CollapsibleNavbar.vue";
import PerformanceDrawer from "./components/PerformanceDrawer.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";
import { useSettings } from "./store/settings";

const settings = useSettings();

const theme = computed(() => {
  if (settings.data.settings.frontend.theme === "dark") {
    document.body.style.backgroundColor = "#121215";
    return darkTheme;
  } else {
    document.body.style.backgroundColor = "white";
    return lightTheme;
  }
});

const backgroundColor = computed(() => {
  if (settings.data.settings.frontend.theme === "dark") {
    return "#121215";
  } else {
    return "#fff";
  }
});

const overrides: GlobalThemeOverrides = {
  common: {
    fontSize: "15px",
    fontWeight: "600",
  },
};
</script>

<style scoped>
.main {
  background-color: v-bind(backgroundColor);
}
</style>
