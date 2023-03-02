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
        @click="accelerate"
        >Accelerate</NButton
      >
    </NSpace>
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
  NSelect,
  NSlider,
  NSpace,
  NStep,
  NSteps,
  type SelectOption,
} from "naive-ui";
import { reactive, ref } from "vue";

const state = useState();
const width = ref(512);
const height = ref(512);
const batchSize = ref(1);
const model = ref("");
const modelOptions: Array<SelectOption> = reactive([]);

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

const accelerate = async () => {
  await fetch(`${serverUrl}/api/generate/generate-aitemplate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model_id: model.value,
      width: width.value,
      height: height.value,
      batch_ize: batchSize.value,
    }),
  });
};
</script>
