<template>
  <div style="margin: 16px">
    <NCard title="Acceleration progress">
      <NSpace vertical justify="center">
        <NSteps>
          <NStep
            title="UNet"
            description="The 'make a better guess' machine (takes a while)"
            :status="state.state.aitBuildStep.unet"
          />
          <NStep
            title="CLIP"
            description="Text encoder (usually quite fast)"
            :status="state.state.aitBuildStep.clip"
          />
          <NStep
            title="VAE"
            description="Upscaler (something in between)"
            :status="state.state.aitBuildStep.vae"
          />
          <NStep
            title="Cleanup"
            description="Get rid of the temporary build files"
            :status="state.state.aitBuildStep.cleanup"
          /> </NSteps></NSpace
    ></NCard>

    <NCard style="margin-top: 16px">
      <!-- Width -->
      <div class="flex-container">
        <p class="slider-label">Width</p>
        <NSlider
          v-model:value="width"
          :min="128"
          :max="2048"
          :step="64"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="width"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="64"
          :min="128"
          :max="2048"
        />
      </div>

      <!-- Height -->
      <div class="flex-container">
        <p class="slider-label">Height</p>
        <NSlider
          v-model:value="height"
          :min="128"
          :max="2048"
          :step="64"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="height"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="64"
          :min="128"
          :max="2048"
        />
      </div>

      <!-- Batch Size -->
      <div class="flex-container">
        <p class="slider-label">Batch Size</p>
        <NSlider
          v-model:value="batchSize"
          :min="1"
          :max="9"
          :step="1"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="batchSize"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="1"
          :min="1"
          :max="9"
        />
      </div>

      <!-- CPU Threads -->
      <div class="flex-container">
        <p class="slider-label">CPU Threads (affects RAM usage)</p>
        <NSlider
          v-model:value="threads"
          :step="1"
          :min="1"
          :max="64"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="threads"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="1"
          :min="1"
          :max="64"
        />
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
        >Accelerate</NButton
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
      @negative-click="accelerate"
    />
  </div>
</template>

<script lang="ts" setup>
import type { ModelEntry } from "@/core/interfaces";
import { serverUrl } from "@/env";
import { useState } from "@/store/state";
import {
  NButton,
  NCard,
  NInputNumber,
  NModal,
  NSelect,
  NSlider,
  NSpace,
  NStep,
  NSteps,
  useMessage,
  type SelectOption,
} from "naive-ui";
import { reactive, ref } from "vue";

const message = useMessage();

const state = useState();
const width = ref(512);
const height = ref(512);
const batchSize = ref(1);
const model = ref("");
const threads = ref(8);
const modelOptions: Array<SelectOption> = reactive([]);

const building = ref(false);
const showUnloadModal = ref(false);

fetch(`${serverUrl}/api/models/avaliable`).then((res) => {
  res.json().then((data: Array<ModelEntry>) => {
    modelOptions.splice(0, modelOptions.length);

    const pyTorch = data.filter((x) => x.backend === "PyTorch");

    if (pyTorch) {
      for (const model of pyTorch) {
        modelOptions.push({
          label: model.name,
          value: model.name,
        });
      }
    }

    if (modelOptions.length > 0) {
      model.value = pyTorch[0].name;
    }
  });
});

const accelerateUnload = async () => {
  try {
    await fetch(`${serverUrl}/api/models/unload-all`, {
      method: "POST",
    });

    showUnloadModal.value = false;
    await accelerate();
  } catch {
    showUnloadModal.value = false;
    message.error("Failed to unload, check the console for more info.");
  }
};

const accelerate = async () => {
  building.value = true;
  await fetch(`${serverUrl}/api/generate/generate-aitemplate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model_id: model.value,
      width: width.value,
      height: height.value,
      batch_size: batchSize.value,
      threads: threads.value,
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
