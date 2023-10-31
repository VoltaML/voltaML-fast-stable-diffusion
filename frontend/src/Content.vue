<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NNotificationProvider placement="bottom-right" :max="3">
    <NLoadingBarProvider>
      <NMessageProvider>
        <div id="background"></div>
        <SecretsHandler />
        <CollapsibleNavbar />
        <TopBar />
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
import {
  CollapsibleNavbar,
  InitHandler,
  LogDrawer,
  PerformanceDrawer,
  SecretsHandler,
  TopBar,
} from "./components";
import routerContainerVue from "./router/router-container.vue";
</script>
