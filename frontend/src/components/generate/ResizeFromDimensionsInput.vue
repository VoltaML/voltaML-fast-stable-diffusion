<!-- eslint-disable vue/no-mutating-props -->
<template>
  <div class="flex-container">
    <p class="slider-label">Resize From (temporary placement)</p>
    <NSwitch v-model:value="show" />
  </div>

  <div v-if="show">
    <div class="flex-container">
      <p class="slider-label">Resize From Width</p>
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
    <div class="flex-container">
      <p class="slider-label">Resize From Height</p>
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
  </div>
</template>

<script setup lang="ts">
import { NInputNumber, NSlider, NSwitch } from "naive-ui";
import type { PropType } from "vue";
import { ref } from "vue";

const show = ref(false);

interface DimensionsObject {
  width: number;
  height: number;
}

const props = defineProps({
  dimensionsObject: {
    type: Object as PropType<DimensionsObject>,
    required: true,
  },
});
</script>
