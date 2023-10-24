<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NNotificationProvider placement="bottom-right" :max="3">
    <NLoadingBarProvider>
      <NMessageProvider>
        <div id="background"></div>
        <SecretsHandlerVue />
        <CollapsileNavbarVue />
        <TopBarVue />
        <InitHandler />
        <routerContainerVue style="margin-top: 52px" />
        <PerformanceDrawer />
        <LogDrawer />
      </NMessageProvider>
    </NLoadingBarProvider>
  </NNotificationProvider>
</template>

<script lang="ts" setup>
import {
  NLoadingBarProvider,
  NMessageProvider,
  NNotificationProvider,
} from "naive-ui";
import CollapsileNavbarVue from "./components/CollapsibleNavbar.vue";
import InitHandler from "./components/InitHandler.vue";
import LogDrawer from "./components/LogDrawer.vue";
import PerformanceDrawer from "./components/PerformanceDrawer.vue";
import SecretsHandlerVue from "./components/SecretsHandler.vue";
import TopBarVue from "./components/TopBar.vue";
import routerContainerVue from "./router/router-container.vue";
</script>
