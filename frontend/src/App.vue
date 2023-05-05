<template>
  <NConfigProvider :theme="theme" :theme-overrides="overrides" class="main">
    <NNotificationProvider placement="bottom-right">
      <NMessageProvider>
        <CollapsileNavbarVue />
        <TopBarVue />
        <routerContainerVue />
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
    return darkTheme;
  } else {
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
