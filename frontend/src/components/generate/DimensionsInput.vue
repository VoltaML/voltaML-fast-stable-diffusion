<!-- eslint-disable vue/no-mutating-props -->
<template>
  <div class="flex-container" v-if="settings.data.settings.aitDim.width">
    <p class="slider-label">Width</p>
    <NSlider
      v-model:value="props.dimensionsObject.width"
      :min="settings.data.settings.aitDim.width[0]"
      :max="settings.data.settings.aitDim.width[1]"
      :step="64"
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
    />
    <NInputNumber
      v-model:value="props.dimensionsObject.height"
      size="small"
      style="min-width: 96px; width: 96px"
      :step="1"
    />
  </div>
</template>

<script setup lang="ts">
import { NInputNumber, NSlider } from "naive-ui";
import type { PropType } from "vue";
import { useSettings } from "../../store/settings";

interface DimensionsObject {
  width: number;
  height: number;
}

const settings = useSettings();

const props = defineProps({
  dimensionsObject: {
    type: Object as PropType<DimensionsObject>,
    required: true,
  },
});
</script>
