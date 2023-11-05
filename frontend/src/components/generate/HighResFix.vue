<template>
  <NCard title="Highres fix">
    <div class="flex-container">
      <div class="slider-label">
        <p>Enabled</p>
      </div>
      <NSwitch v-model:value="global.state.txt2img.highres" />
    </div>

    <NSpace vertical class="left-container" v-if="global.state.txt2img.highres">
      <!-- Mode -->
      <div class="flex-container">
        <div class="slider-label">
          <p>Mode</p>
        </div>
        <NSelect
          v-model:value="settings.data.settings.flags.highres.mode"
          :options="[
            { label: 'Latent', value: 'latent' },
            { label: 'Image', value: 'image' },
          ]"
        />
      </div>

      <!-- Mode options -->
      <div v-if="settings.data.settings.flags.highres.mode === 'image'">
        <div class="flex-container">
          <p class="slider-label">Upscaler</p>
          <NSelect
            v-model:value="settings.data.settings.flags.highres.image_upscaler"
            size="small"
            style="flex-grow: 1"
            filterable
            :options="imageUpscalerOptions"
          />
        </div>
      </div>
      <div v-else>
        <div class="flex-container">
          <p class="slider-label">Antialiased</p>
          <NSwitch
            v-model:value="settings.data.settings.flags.highres.antialiased"
          />
        </div>

        <div class="flex-container">
          <p class="slider-label">Latent Mode</p>
          <NSelect
            v-model:value="
              settings.data.settings.flags.highres.latent_scale_mode
            "
            size="small"
            style="flex-grow: 1"
            filterable
            :options="latentUpscalerOptions"
          />
        </div>
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
          v-model:value="settings.data.settings.flags.highres.steps"
          :min="5"
          :max="300"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.highres.steps"
          size="small"
          style="min-width: 96px; width: 96px"
        />
      </div>

      <!-- Scale -->
      <div class="flex-container">
        <p class="slider-label">Scale</p>
        <NSlider
          v-model:value="settings.data.settings.flags.highres.scale"
          :min="1"
          :max="8"
          :step="0.1"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.highres.scale"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="0.1"
        />
      </div>

      <!-- Denoising strength -->
      <div class="flex-container">
        <p class="slider-label">Strength</p>
        <NSlider
          v-model:value="settings.data.settings.flags.highres.strength"
          :min="0.1"
          :max="0.9"
          :step="0.05"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.highres.strength"
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
import { upscalerOptions, useSettings } from "@/store/settings";
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
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { computed } from "vue";

const settings = useSettings();
const global = useState();

const imageUpscalerOptions = computed<SelectMixedOption[]>(() => {
  const localModels = global.state.models
    .filter(
      (model) =>
        model.backend === "Upscaler" &&
        !(
          upscalerOptions
            .map((option: SelectMixedOption) => option.label)
            .indexOf(model.name) !== -1
        )
    )
    .map((model) => ({
      label: model.name,
      value: model.path,
    }));

  return [...upscalerOptions, ...localModels];
});

const latentUpscalerOptions: SelectMixedOption[] = [
  { label: "Nearest", value: "nearest" },
  { label: "Nearest exact", value: "nearest-exact" },
  { label: "Area", value: "area" },
  { label: "Bilinear", value: "bilinear" },
  { label: "Bicubic", value: "bicubic" },
  {
    label: "Bislerp (Original, slow)",
    value: "bislerp-original",
  },
  {
    label: "Bislerp (Tortured, fast)",
    value: "bislerp-tortured",
  },
];
</script>
