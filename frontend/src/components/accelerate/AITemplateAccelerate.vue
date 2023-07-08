<template>
  <div style="margin: 16px">
    <NCard title="Acceleration progress (around 20 minutes)">
      <NSpace vertical justify="center">
        <NSteps>
          <NStep title="UNet" :status="global.state.aitBuildStep.unet" />
          <NStep
            title="ControlNet UNet"
            :status="global.state.aitBuildStep.controlnet_unet"
          />
          <NStep title="CLIP" :status="global.state.aitBuildStep.clip" />
          <NStep title="VAE" :status="global.state.aitBuildStep.vae" />
          <NStep
            title="Cleanup"
            :status="global.state.aitBuildStep.cleanup"
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
import { computed, ref } from "vue";

const message = useMessage();
const global = useState();

const width = ref(512);
const height = ref(512);
const batchSize = ref(1);
const model = ref("");
const threads = ref(8);

const building = ref(false);
const showUnloadModal = ref(false);

const modelOptions = computed(() => {
  const options: SelectOption[] = [];
  for (const model of global.state.models) {
    if (
      model.backend === "PyTorch" &&
      model.valid &&
      !model.name.endsWith(".safetensors") &&
      !model.name.endsWith(".ckpt")
    ) {
      options.push({
        label: model.name,
        value: model.path,
      });
    }
  }
  return options;
});

model.value = modelOptions.value[0]?.value?.toString() ?? "";

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
  showUnloadModal.value = false;
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
