<template>
  <NConfigProvider :theme="darkTheme">
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
          :default-height="0"
        >
          <NDrawerContent>{{ glob.state.drawer_content }} </NDrawerContent>
        </NDrawer>
      </NMessageProvider>
    </NNotificationProvider>
  </NConfigProvider>
</template>

<script setup lang="ts">
import {
  NConfigProvider,
  NDrawer,
  NDrawerContent,
  NMessageProvider,
  NNotificationProvider,
  darkTheme,
} from "naive-ui";
import CollapsileNavbarVue from "./components/CollapsileNavbar.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";
import { useState } from "./store/state";

const glob = useState();
</script>
