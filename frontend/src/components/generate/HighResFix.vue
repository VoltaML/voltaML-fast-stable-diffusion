<template>
  <div class="flex-container">
    <div class="slider-label">
      <p>Enabled</p>
    </div>
    <NSwitch v-model:value="target[props.tab].highres.enabled" />
  </div>

  <NSpace
    vertical
    class="left-container"
    v-if="target[props.tab].highres.enabled"
  >
    <!-- Mode -->
    <div class="flex-container">
      <div class="slider-label">
        <p>Mode</p>
      </div>
      <NSelect
        v-model:value="target[props.tab].highres.mode"
        :options="[
          { label: 'Latent', value: 'latent' },
          { label: 'Image', value: 'image' },
        ]"
      />
    </div>

    <!-- Mode options -->
    <div v-if="target[props.tab].highres.mode === 'image'">
      <div class="flex-container">
        <p class="slider-label">Upscaler</p>
        <NSelect
          v-model:value="target[props.tab].highres.image_upscaler"
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
        <NSwitch v-model:value="target[props.tab].highres.antialiased" />
      </div>

      <div class="flex-container">
        <p class="slider-label">Latent Mode</p>
        <NSelect
          v-model:value="target[props.tab].highres.latent_scale_mode"
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
        <b class="highlight">We recommend using 20-50 steps for most images.</b>
      </NTooltip>
      <NSlider
        v-model:value="target[props.tab].highres.steps"
        :min="5"
        :max="300"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="target[props.tab].highres.steps"
        size="small"
        style="min-width: 96px; width: 96px"
      />
    </div>

    <!-- Scale -->
    <div class="flex-container">
      <p class="slider-label">Scale</p>
      <NSlider
        v-model:value="target[props.tab].highres.scale"
        :min="1"
        :max="8"
        :step="0.1"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="target[props.tab].highres.scale"
        size="small"
        style="min-width: 96px; width: 96px"
        :step="0.1"
      />
    </div>

    <!-- Denoising strength -->
    <div class="flex-container">
      <p class="slider-label">Strength</p>
      <NSlider
        v-model:value="target[props.tab].highres.strength"
        :min="0.1"
        :max="0.9"
        :step="0.05"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="target[props.tab].highres.strength"
        size="small"
        style="min-width: 96px; width: 96px"
        :min="0.1"
        :max="0.9"
        :step="0.05"
      />
    </div>
  </NSpace>
</template>

<script setup lang="ts">
import type { ISettings } from "@/settings";
import { upscalerOptions, useSettings } from "@/store/settings";
import { useState } from "@/store/state";
import {
  NInputNumber,
  NSelect,
  NSlider,
  NSpace,
  NSwitch,
  NTooltip,
} from "naive-ui";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { computed, type PropType } from "vue";
import type { InferenceTabs } from "../../types";

const settings = useSettings();
const global = useState();
const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
  target: {
    type: String as PropType<"settings" | "defaultSettings">,
    required: false,
    default: "settings",
  },
});

const target = computed<ISettings>(() => {
  if (props.target === "settings") {
    return settings.data.settings;
  }

  return settings.defaultSettings;
});

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
  { label: "Bislerp", value: "bislerp" },
];
</script>
