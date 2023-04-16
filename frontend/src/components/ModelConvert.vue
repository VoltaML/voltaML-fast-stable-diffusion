<template>
  <div style="margin: 16px">
    <NCard style="margin-top: 16px">
      <!-- FP32 -->
      <div class="flex-container">
        <p class="slider-label">FP32</p>
        <NSwitch v-model:value="use_fp32" />
      </div>

      <!-- Safetensors -->
      <div class="flex-container">
        <p class="slider-label">Output in safetensors format</p>
        <NSwitch v-model:value="safetensors" />
      </div>

      <!-- Model select -->
      <div class="flex-container">
        <p class="slider-label">Model</p>
        <NSelect
          v-model:value="model"
          :options="modelOptions"
          style="margin-right: 12px"
        />
      </div>
    </NCard>

    <NSpace vertical justify="center" style="width: 100%" align="center">
      <NButton
        style="margin-top: 16px; padding: 0 92px"
        type="success"
        ghost
        :loading="building"
        :disabled="building || modelOptions.length === 0"
        @click="showUnloadModal = true"
        >Convert</NButton
      >
    </NSpace>

    <NModal
      v-model:show="showUnloadModal"
      preset="dialog"
      title="Unload other models"
      width="400px"
      :closable="false"
      :show-close="false"
      content="Acceleration can be done with the other models loaded as well, but it will take a lot of resources. It is recommended to unload the other models before accelerating. Do you want to unload the other models?"
      positive-text="Unload models"
      negative-text="Keep models"
      @positive-click="accelerateUnload"
      @negative-click="convert"
    />
  </div>
</template>

<script lang="ts" setup>
import type { ModelEntry } from "@/core/interfaces";
import { serverUrl } from "@/env";
import {
  NButton,
  NCard,
  NModal,
  NSelect,
  NSpace,
  NSwitch,
  useMessage,
  type SelectOption,
} from "naive-ui";
import { reactive, ref } from "vue";

const message = useMessage();

const model = ref("");
const modelOptions: Array<SelectOption> = reactive([]);

const building = ref(false);
const use_fp32 = ref(false);
const safetensors = ref(false);
const showUnloadModal = ref(false);

fetch(`${serverUrl}/api/models/available`).then((res) => {
  res.json().then((data: Array<ModelEntry>) => {
    modelOptions.splice(0, modelOptions.length);

    const pyTorch = data.filter((x) => x.backend === "PyTorch");

    if (pyTorch) {
      for (const model of pyTorch) {
        modelOptions.push({
          label: model.name,
          value: model.name,
          disabled: !model.valid,
        });
      }
    }
  });
});

const accelerateUnload = async () => {
  try {
    await fetch(`${serverUrl}/api/models/unload-all`, {
      method: "POST",
    });

    showUnloadModal.value = false;
    await convert();
  } catch {
    showUnloadModal.value = false;
    message.error("Failed to unload, check the console for more info.");
  }
};

const convert = async () => {
  showUnloadModal.value = false;
  building.value = true;
  await fetch(`${serverUrl}/api/generate/convert-model`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: model.value,
      use_fp32: use_fp32.value,
      safetensors: safetensors.value,
    }),
  })
    .then(() => {
      building.value = false;
    })
    .catch(() => {
      building.value = false;
      message.error("Failed to accelerate, check the console for more info.");
    });
};
</script>
