<template>
  <NSpace justify="end" inline align="center" class="top-bar">
    <NButton
      :type="websocketState.color"
      quaternary
      icon-placement="left"
      :render-icon="syncIcon"
      :loading="websocketState.loading"
      @click="websocketState.ws_open"
      >{{ websocketState.text }}</NButton
    >
  </NSpace>
</template>

<script lang="ts" setup>
import { useWebsocket } from "@/store/websocketState";
import { SyncSharp } from "@vicons/ionicons5";
import { NButton, NSpace } from "naive-ui";
import { h } from "vue";

const websocketState = useWebsocket();

const syncIcon = () => {
  return h(SyncSharp);
};
</script>

<style scoped>
.top-bar {
  border-bottom: #505050 1px solid;
  padding-top: 10px;
  padding-bottom: 10px;
  width: 100%;
  background-color: rgb(24, 24, 28, 0.6);
}

.logo {
  margin-right: 16px;
  margin-left: 16px;
}
</style>
