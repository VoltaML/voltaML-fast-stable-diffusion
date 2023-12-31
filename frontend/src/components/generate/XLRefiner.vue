<template>
  <NCard title="SDXL Refiner" class="generate-extra-card">
    <div class="flex-container">
      <div class="slider-label">
        <p>Enabled</p>
      </div>
      <NSwitch v-model:value="global.state.txt2img.refiner" />
    </div>
    <NSpace vertical class="left-container" v-if="global.state.txt2img.refiner">
      <!-- Refiner model -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Refiner model</p>
          </template>
          The SDXL-Refiner model to use for this step of diffusion.
          <b class="highlight">
            Generally, the refiner that came with your model is bound to
            generate the best results.
          </b>
        </NTooltip>
        <NSelect
          :options="refinerModels"
          placeholder="None"
          @update:value="onRefinerChange"
          :value="
            settings.data.settings.flags.refiner.model !== null
              ? settings.data.settings.flags.refiner.model
              : ''
          "
        />
      </div>

      <!-- Steps -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Steps</p>
          </template>
          Number of steps to take in the diffusion process. Higher values will
          result in more detailed images but will take longer to generate. There
          is also a point of diminishing returns around 100 steps.
          <b class="highlight"
            >We recommend using 20-50 steps for most images.</b
          >
        </NTooltip>
        <NSlider
          v-model:value="settings.data.settings.flags.refiner.steps"
          :min="5"
          :max="300"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.refiner.steps"
          size="small"
          style="min-width: 96px; width: 96px"
        />
      </div>

      <!-- Aesthetic score -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Aesthetic Score</p>
          </template>
          Generally higher numbers will produce "more professional" images.
          <b class="highlight">Generally best to keep it around 6.</b>
        </NTooltip>
        <NSlider
          v-model:value="settings.data.settings.flags.refiner.aesthetic_score"
          :min="0"
          :max="10"
          :step="0.5"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.refiner.aesthetic_score"
          :min="0"
          :max="10"
          :step="0.25"
          size="small"
          style="min-width: 96px; width: 96px"
        />
      </div>

      <!-- Negative aesthetic score -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Negative Aesthetic Score</p>
          </template>
          Makes sense to keep this lower than aesthetic score.
          <b class="highlight">Generally best to keep it around 3.</b>
        </NTooltip>
        <NSlider
          v-model:value="
            settings.data.settings.flags.refiner.negative_aesthetic_score
          "
          :min="0"
          :max="10"
          :step="0.5"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="
            settings.data.settings.flags.refiner.negative_aesthetic_score
          "
          :min="0"
          :max="10"
          :step="0.25"
          size="small"
          style="min-width: 96px; width: 96px"
        />
      </div>

      <div class="flex-container">
        <p class="slider-label">Strength</p>
        <NSlider
          v-model:value="settings.data.settings.flags.refiner.strength"
          :min="0.1"
          :max="0.9"
          :step="0.05"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.refiner.strength"
          size="small"
          style="min-width: 96px; width: 96px"
          :min="0.1"
          :max="0.9"
          :step="0.05"
        />
      </div>
    </NSpace>
  </NCard>
</template>

<script setup lang="ts">
import { useSettings } from "@/store/settings";
import { useState } from "@/store/state";
import {
  NCard,
  NInputNumber,
  NSelect,
  NSlider,
  NSpace,
  NSwitch,
  NTooltip,
} from "naive-ui";
import { computed } from "vue";

const settings = useSettings();
const global = useState();

const refinerModels = computed(() => {
  return global.state.models
    .filter((model) => model.type === "SDXL")
    .map((model) => {
      return {
        label: model.name,
        value: model.name,
      };
    });
});

async function onRefinerChange(modelStr: string) {
  settings.data.settings.flags.refiner.model = modelStr;
}
</script>
