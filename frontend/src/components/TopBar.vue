<template>
  <div class="top-bar">
    <!-- Progress bar -->
    <div class="progress-container">
      <NProgress
        type="line"
        :percentage="global.state.progress"
        indicator-placement="outside"
        :processing="global.state.progress < 100 && global.state.progress > 0"
        color="#63e2b7"
        :show-indicator="true"
      >
        {{ global.state.current_step }} / {{ global.state.total_steps }}
      </NProgress>
    </div>
    <NSpace inline justify="end" align="center">
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
  </div>
</template>

<script lang="ts" setup>
import { useWebsocket } from "@/store/websockets";
import { SyncSharp } from "@vicons/ionicons5";
import { NButton, NProgress, NSpace } from "naive-ui";
import { h } from "vue";
import { useState } from "../store/state";

const websocketState = useWebsocket();
const global = useState();

const syncIcon = () => {
  return h(SyncSharp);
};
</script>

<style scoped>
.progress-container {
  margin: 12px;
  flex-grow: 1;
  width: 400px;
}
.top-bar {
  display: inline-flex;
  align-items: center;
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
