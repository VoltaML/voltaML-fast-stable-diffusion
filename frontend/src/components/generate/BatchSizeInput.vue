<template>
  <div class="flex-container" v-if="settings.data.settings.aitDim.batch_size">
    <NTooltip style="max-width: 600px">
      <template #trigger>
        <p class="slider-label">Batch Size</p>
      </template>
      Number of images to generate in paralel.
    </NTooltip>
    <NSlider
      v-model:value="props.batchSizeObject.batch_size"
      :min="settings.data.settings.aitDim.batch_size[0]"
      :max="settings.data.settings.aitDim.batch_size[1]"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="props.batchSizeObject.batch_size"
      size="small"
      :min="settings.data.settings.aitDim.batch_size[0]"
      :max="settings.data.settings.aitDim.batch_size[1]"
      style="min-width: 96px; width: 96px"
    />
  </div>
  <div class="flex-container" v-else>
    <NTooltip style="max-width: 600px">
      <template #trigger>
        <p class="slider-label">Batch Size</p>
      </template>
      Number of images to generate in paralel.
    </NTooltip>
    <NSlider
      v-model:value="props.batchSizeObject.batch_size"
      :min="1"
      :max="9"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="props.batchSizeObject.batch_size"
      size="small"
      style="min-width: 96px; width: 96px"
    />
  </div>
</template>

<script setup lang="ts">
import { NInputNumber, NSlider, NTooltip } from "naive-ui";
import type { PropType } from "vue";
import { useSettings } from "../../store/settings";

interface DimensionsObject {
  batch_size: number;
}

const settings = useSettings();

const props = defineProps({
  batchSizeObject: {
    type: Object as PropType<DimensionsObject>,
    required: true,
  },
});
</script>
