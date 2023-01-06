<template>
  <NConfigProvider :theme="darkTheme" :theme-overrides="themeOverrides">
    <NNotificationProvider placement="bottom-right">
      <WebSocketHandler>
        <NMessageProvider>
          <CollapsileNavbarVue />
          <TopBarVue />
          <routerContainerVue />
        </NMessageProvider>
      </WebSocketHandler>
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
import WebSocketHandler from "./websockets/WebSocketHandler.vue";

const themeOverrides: GlobalThemeOverrides = {};
</script>
