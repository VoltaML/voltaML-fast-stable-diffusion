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
  useThemeVars,
  type GlobalThemeOverrides,
} from "naive-ui";
import CollapsileNavbarVue from "./components/CollapsileNavbar.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";

const theme = useThemeVars();
theme.value.scrollbarHeight = "8px";

const themeOverrides: GlobalThemeOverrides = {
  Scrollbar: {
    color: "#63e2b7b0",
    colorHover: "#63e2b7",
  },
};
</script>
