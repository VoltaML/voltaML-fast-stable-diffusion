<template>
  <NConfigProvider :theme="darkTheme" :theme-overrides="overrides">
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
  type GlobalThemeOverrides,
} from "naive-ui";
import CollapsileNavbarVue from "./components/CollapsibleNavbar.vue";
import PerformanceDrawer from "./components/PerformanceDrawer.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";

const overrides: GlobalThemeOverrides = {
  common: {
    fontSize: "10",
    fontWeight: "600",
  },
};
</script>
