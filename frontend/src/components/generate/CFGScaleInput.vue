<template>
  <!-- CFG Scale -->
  <div class="flex-container">
    <NTooltip style="max-width: 600px">
      <template #trigger>
        <p class="slider-label">CFG Scale</p>
      </template>
      Guidance scale indicates how close should the model stay to the prompt.
      Higher values might be exactly what you want, but generated images might
      have some artifacts. Lower values give the model more freedom, and
      therefore might produce more coherent/less-artifacty images, but wouldn't
      follow the prompt as closely.
      <b class="highlight">We recommend using 3-15 for most images.</b>
    </NTooltip>
    <NSlider
      v-model:value="settings.data.settings[props.tab].cfg_scale"
      :min="1"
      :max="cfgMax"
      :step="0.5"
      style="margin-right: 12px"
    />
    <NInputNumber
      v-model:value="settings.data.settings[props.tab].cfg_scale"
      size="small"
      style="min-width: 96px; width: 96px"
      :min="1"
      :max="cfgMax"
      :step="0.5"
    />
  </div>
</template>

<script lang="ts" setup>
import { NTooltip, NSlider, NInputNumber } from "naive-ui";
import { computed, type PropType } from "vue";
import type { InferenceTabs } from "@/types";
import { useSettings } from "../../store/settings";

const settings = useSettings();

const cfgMax = computed(() => {
  var scale = 30;
  return (
    scale +
    Math.max(
      settings.defaultSettings.api.apply_unsharp_mask ? 15 : 0,
      settings.defaultSettings.api.cfg_rescale_threshold == "off" ? 0 : 30
    )
  );
});

const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
});
</script>
