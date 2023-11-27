<!-- eslint-disable vue/no-mutating-props -->
<template>
  <div class="flex-container" v-if="settings.data.settings.aitDim.width">
    <p class="slider-label">Width</p>
    <NSlider
      v-model:value="props.dimensionsObject.width"
      :min="settings.data.settings.aitDim.width[0]"
      :max="settings.data.settings.aitDim.width[1]"
      :step="64"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="props.dimensionsObject.width"
      size="small"
      style="min-width: 96px; width: 96px"
      :min="settings.data.settings.aitDim.width[0]"
      :max="settings.data.settings.aitDim.width[1]"
      :step="64"
    />
  </div>
  <div class="flex-container" v-else>
    <p class="slider-label">Width</p>
    <NSlider
      v-model:value="props.dimensionsObject.width"
      :min="128"
      :max="2048"
      :step="1"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="props.dimensionsObject.width"
      size="small"
      style="min-width: 96px; width: 96px"
      :step="1"
    />
  </div>
  <div class="flex-container" v-if="settings.data.settings.aitDim.height">
    <p class="slider-label">Height</p>
    <NSlider
      v-model:value="props.dimensionsObject.height"
      :min="settings.data.settings.aitDim.height[0]"
      :max="settings.data.settings.aitDim.height[1]"
      :step="64"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="props.dimensionsObject.height"
      size="small"
      style="min-width: 96px; width: 96px"
      :min="settings.data.settings.aitDim.height[0]"
      :max="settings.data.settings.aitDim.height[1]"
      :step="64"
    />
  </div>
  <div class="flex-container" v-else>
    <p class="slider-label">Height</p>
    <NSlider
      v-model:value="props.dimensionsObject.height"
      :min="128"
      :max="2048"
      :step="1"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="props.dimensionsObject.height"
      size="small"
      style="min-width: 96px; width: 96px"
      :step="1"
    />
  </div>
  <div
    v-if="
      props.dimensionsObject.width * props.dimensionsObject.height >=
        768 * 768 || settings.defaultSettings.flags.deepshrink.enabled
    "
  >
    <div class="flex-container">
      <p class="slider-label">Enable Deepshrink</p>
      <NSwitch
        v-model:value="settings.defaultSettings.flags.deepshrink.enabled"
      />
    </div>
    <div v-if="settings.defaultSettings.flags.deepshrink.enabled">
      <div class="flex-container">
        <p class="slider-label">First layer</p>
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.deepshrink.depth_1"
          :max="4"
          :min="1"
          :step="1"
        />
        <div></div>
        <p class="slider-label">Stop at</p>
        <NSlider
          v-model:value="settings.defaultSettings.flags.deepshrink.stop_at_1"
          :min="0.05"
          :max="1.0"
          :step="0.05"
        />
        <div></div>
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.deepshrink.stop_at_1"
          :max="1"
          :min="0.05"
          :step="0.05"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">Second layer</p>
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.deepshrink.depth_2"
          :max="4"
          :min="1"
          :step="1"
        />
        <div></div>
        <p class="slider-label">Stop at</p>
        <NSlider
          v-model:value="settings.defaultSettings.flags.deepshrink.stop_at_2"
          :min="0.05"
          :max="1.0"
          :step="0.05"
        />
        <div></div>
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.deepshrink.stop_at_2"
          :max="1"
          :min="0.05"
          :step="0.05"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">Scale</p>
        <NSlider
          v-model:value="settings.defaultSettings.flags.deepshrink.base_scale"
          :min="0.05"
          :max="1.0"
          :step="0.05"
        />
        <div></div>
        <NInputNumber
          v-model:value="settings.defaultSettings.flags.deepshrink.base_scale"
          :max="1"
          :min="0.05"
          :step="0.05"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">Latent scaler</p>
        <NSelect
          v-model:value="settings.defaultSettings.flags.deepshrink.scaler"
          filterable
          :options="latentUpscalerOptions"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">Early out</p>
        <NSwitch
          v-model:value="settings.defaultSettings.flags.deepshrink.early_out"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { NInputNumber, NSlider, NSwitch, NSelect } from "naive-ui";
import type { PropType } from "vue";
import { useSettings } from "../../store/settings";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";

interface DimensionsObject {
  width: number;
  height: number;
}

const latentUpscalerOptions: SelectMixedOption[] = [
  { label: "Nearest", value: "nearest" },
  { label: "Nearest exact", value: "nearest-exact" },
  { label: "Area", value: "area" },
  { label: "Bilinear", value: "bilinear" },
  { label: "Bicubic", value: "bicubic" },
  { label: "Bislerp", value: "bislerp" },
];

const settings = useSettings();

const props = defineProps({
  dimensionsObject: {
    type: Object as PropType<DimensionsObject>,
    required: true,
  },
});
</script>
