<template>
  <!-- Self Attention Scale -->
  <div
    class="flex-container"
    v-if="settings.data.settings.model?.backend === 'PyTorch'"
  >
    <NTooltip style="max-width: 600px">
      <template #trigger>
        <p class="slider-label">Self Attention Scale</p>
      </template>
      If self attention is >0, SAG will guide the model and improve the quality
      of the image at the cost of speed. Higher values will follow the guidance
      more closely, which can lead to better, more sharp and detailed outputs.
    </NTooltip>

    <NSlider
      v-model:value="settings.data.settings[props.tab].self_attention_scale"
      :min="0"
      :max="1"
      :step="0.05"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="settings.data.settings[props.tab].self_attention_scale"
      size="small"
      style="min-width: 96px; width: 96px"
      :step="0.05"
    />
  </div>
</template>

<script lang="ts" setup>
import { NTooltip, NSlider, NInputNumber } from "naive-ui";
import { useSettings } from "@/store/settings";
import type { PropType } from "vue";
import type { InferenceTabs } from "@/types";

const settings = useSettings();

const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
});
</script>
