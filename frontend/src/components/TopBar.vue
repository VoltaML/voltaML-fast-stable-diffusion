<template>
  <div class="top-bar">
    <NSelect
      style="max-width: 250px; padding-left: 12px; padding-right: 12px"
      :options="modelOptions"
      @update:value="onModelChange"
      :loading="modelsLoading"
      placeholder="Select model"
      default-value="none:PyTorch"
      :value="conf.data.settings.model"
      :consistent-menu-width="false"
      filterable
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
import type { ModelEntry } from "@/core/interfaces";
import { serverUrl } from "@/env";
import { useWebsocket } from "@/store/websockets";
import { ReloadOutline, SyncSharp } from "@vicons/ionicons5";
import { NButton, NIcon, NProgress, NSelect, NSpace } from "naive-ui";
import type { SelectGroupOption } from "naive-ui/es/select/src/interface";
import { h, reactive, ref } from "vue";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";

const websocketState = useWebsocket();
const global = useState();
const conf = useSettings();
const modelsLoading = ref(false);

function refreshModels() {
  modelsLoading.value = true;
  fetch(`${serverUrl}/api/models/avaliable`).then((res) => {
    res.json().then((data: Array<ModelEntry>) => {
      // add all the strings from the list to model options
      modelOptions.splice(0, modelOptions.length);

      const pytorchGroup: SelectGroupOption = {
        type: "group",
        label: "PyTorch",
        key: "pytorch",
        children: [],
      };

      const tensorrtGroup: SelectGroupOption = {
        type: "group",
        label: "TensorRT",
        key: "tensorrt",
        children: [],
      };

      const aitemplatesGroup: SelectGroupOption = {
        type: "group",
        label: "AITemplate",
        key: "aitemplate",
        children: [],
      };

      data.forEach((item) => {
        if (item.backend === "PyTorch") {
          pytorchGroup.children?.push({
            label: item.name,
            value: item.path + ":" + item.backend,
            style: item.valid
              ? "color: #eb7028"
              : "text-decoration: line-through",
          });
        } else if (item.backend === "TensorRT") {
          tensorrtGroup.children?.push({
            label: item.name,
            value: item.path + ":" + item.backend,
            style: item.valid
              ? "color: #28eb6c"
              : "text-decoration: line-through",
          });
        } else if (item.backend === "AITemplate") {
          aitemplatesGroup.children?.push({
            label: item.name,
            value: item.path + ":" + item.backend,
            style: item.valid
              ? "color: #48bdf0"
              : "text-decoration: line-through",
          });
        }
      });

      modelOptions.push(tensorrtGroup);
      modelOptions.push(aitemplatesGroup);
      modelOptions.push(pytorchGroup);
      modelOptions.push(defaultOptions);

      modelsLoading.value = false;
    });
  });
}

async function onModelChange(value: string) {
  const model = value.split(":")[0];
  const x = value.split(":")[1];

  if (x !== "PyTorch" && x !== "TensorRT" && x !== "AITemplate") {
    throw new Error("Invalid backend");
  }

  const backend: "PyTorch" | "TensorRT" | "AITemplate" = x;

  await fetch(`${serverUrl}/api/models/unload-all`, {
    method: "POST",
  });

  if (model === "none") {
    conf.data.settings.model = value;
    return;
  }

  conf.data.settings.backend = backend;

  const load_url = new URL(`${serverUrl}/api/models/load`);
  const params = { model: model, backend: backend };
  load_url.search = new URLSearchParams(params).toString();

  modelsLoading.value = true;

  await fetch(load_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  }).catch(() => {
    modelsLoading.value = false;
  });

  modelsLoading.value = false;

  conf.data.settings.model = model;
}

const syncIcon = () => {
  return h(SyncSharp);
};

const defaultOptions: SelectGroupOption = {
  type: "group",
  label: "Unload",
  key: "unload",
  children: [
    {
      label: "No model selected",
      value: "none:PyTorch",
    },
  ],
};
const modelOptions: Array<SelectGroupOption> = reactive([defaultOptions]);

websocketState.onConnectedCallbacks.push(() => {
  refreshModels();
});
if (websocketState.readyState === "OPEN") {
  refreshModels();
}
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
  height: 32px;
}

.logo {
  margin-right: 16px;
  margin-left: 16px;
}
</style>
