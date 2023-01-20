<template>
  <div class="top-bar">
    <NSelect
      style="max-width: 250px; padding-left: 12px; padding-right: 12px"
      :options="modelOptions"
      @update:value="onModelChange"
      :loading="modelsLoading"
      placeholder="Select model"
      default-value="none"
    />
    <NButton quaternary circle type="default" @click="refreshModels">
      <template #icon>
        <NIcon>
          <ReloadOutline />
        </NIcon>
      </template>
    </NButton>

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
import { serverUrl } from "@/env";
import { useWebsocket } from "@/store/websockets";
import { ReloadOutline, SyncSharp } from "@vicons/ionicons5";
import { NButton, NIcon, NProgress, NSelect, NSpace } from "naive-ui";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { h, reactive, ref } from "vue";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";

const websocketState = useWebsocket();
const global = useState();
const conf = useSettings();
const modelsLoading = ref(true);

function refreshModels() {
  modelsLoading.value = true;
  fetch(`${serverUrl}/api/models/avaliable`).then((res) => {
    res.json().then((data: Array<string>) => {
      // add all the strings from the list to model options
      modelOptions.splice(0, modelOptions.length);
      modelOptions.push(...defaultOptions);
      data.forEach((item) => {
        modelOptions.push({
          label: item.split("/")[1],
          value: item,
        });
      });

      modelsLoading.value = false;
    });
  });
}

async function onModelChange(value: string) {
  await fetch(`${serverUrl}/api/models/unload-all`, {
    method: "POST",
  });

  if (value === "none") {
    conf.data.settings.model = value;
    return;
  }

  const load_url = new URL(`${serverUrl}/api/models/load`);
  const params = { model: value, backend: conf.data.settings.backend };
  load_url.search = new URLSearchParams(params).toString();

  await fetch(load_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  conf.data.settings.model = value;
}

const syncIcon = () => {
  return h(SyncSharp);
};

const defaultOptions: SelectMixedOption[] = [
  {
    label: "No model selected",
    value: "none",
  },
];
const modelOptions: SelectMixedOption[] = reactive([...defaultOptions]);

refreshModels();
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
