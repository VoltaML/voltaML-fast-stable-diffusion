<template>
  <NConfigProvider :theme="darkTheme" :theme-overrides="themeOverrides">
    <NNotificationProvider placement="bottom-right">
      <NMessageProvider>
        <CollapsileNavbarVue />
        <TopBarVue />
        <routerContainerVue />
      </NMessageProvider>
    </NNotificationProvider>
  </NConfigProvider>
</template>

<script setup lang="ts">
import {
  darkTheme,
  NConfigProvider,
  NMessageProvider,
  NNotificationProvider,
  type GlobalThemeOverrides,
} from "naive-ui";
import CollapsileNavbarVue from "./components/CollapsileNavbar.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";

const themeOverrides: GlobalThemeOverrides = {};
</script>
