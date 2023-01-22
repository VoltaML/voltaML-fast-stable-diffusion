<template>
  <NConfigProvider :theme="darkTheme" :theme-overrides="themeOverrides">
    <NNotificationProvider placement="bottom-right">
      <NMessageProvider>
        <CollapsileNavbarVue />
        <TopBarVue />
        <routerContainerVue />
        <NDrawer
          placement="bottom"
          :mask-closable="true"
          :resizable="true"
          :show="true"
          :trap-focus="false"
          :auto-focus="false"
          :block-scroll="false"
          :show-mask="false"
          :default-height="400"
        >
          <NDrawerContent>{{ glob.state.drawer_content }} </NDrawerContent>
        </NDrawer>
      </NMessageProvider>
    </NNotificationProvider>
  </NConfigProvider>
</template>

<script setup lang="ts">
import {
  darkTheme,
  NConfigProvider,
  NDrawer,
  NDrawerContent,
  NMessageProvider,
  NNotificationProvider,
  type GlobalThemeOverrides,
} from "naive-ui";
import CollapsileNavbarVue from "./components/CollapsileNavbar.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";
import { useState } from "./store/state";

const glob = useState();

const themeOverrides: GlobalThemeOverrides = {
  Scrollbar: {
    color: "#63e2b7b0",
    colorHover: "#63e2b7",
  },
};
</script>
