<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NNotificationProvider placement="bottom-right" :max="2">
    <NLoadingBarProvider>
      <NMessageProvider>
        <div id="background"></div>
        <SecretsHandler />
        <CollapsibleNavbar />
        <TopBar />
        <InitHandler />
        <routerContainerVue class="router-container" />
        <PerformanceDrawer />
        <LogDrawer />
        <SettingsDiffResolver />
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
  SettingsDiffResolver,
  TopBar,
} from "./components";
import routerContainerVue from "./router/router-container.vue";
</script>

<style>
.router-container {
  margin-top: 52px;
}
</style>
