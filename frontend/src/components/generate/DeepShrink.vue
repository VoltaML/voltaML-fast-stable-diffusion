<template>
  <NCard title="Deepshrink" class="generate-extra-card">
    <div class="flex-container">
      <div class="slider-label">
        <p>Enabled</p>
      </div>
      <NSwitch v-model:value="target[props.tab].deepshrink.enabled" />
    </div>

    <div v-if="target[props.tab].deepshrink.enabled">
      <NCard :bordered="false" title="First layer">
        <div class="flex-container space-between">
          <p class="slider-label">Depth</p>
          <NInputNumber
            v-model:value="target[props.tab].deepshrink.depth_1"
            :max="4"
            :min="1"
            :step="1"
          />
        </div>
        <div class="flex-container">
          <p class="slider-label">Stop at</p>
          <NSlider
            v-model:value="target[props.tab].deepshrink.stop_at_1"
            :min="0.05"
            :max="1.0"
            :step="0.05"
            style="margin-right: 12px"
          />
          <NInputNumber
            v-model:value="target[props.tab].deepshrink.stop_at_1"
            :max="1"
            :min="0.05"
            :step="0.05"
          />
        </div>
      </NCard>

      <NCard :bordered="false" title="Second layer">
        <div class="flex-container space-between">
          <p class="slider-label">Depth</p>
          <NInputNumber
            v-model:value="target[props.tab].deepshrink.depth_2"
            :max="4"
            :min="1"
            :step="1"
          />
        </div>
        <div class="flex-container">
          <p class="slider-label">Stop at</p>
          <NSlider
            v-model:value="target[props.tab].deepshrink.stop_at_2"
            :min="0.05"
            :max="1.0"
            :step="0.05"
          />
          <NInputNumber
            v-model:value="target[props.tab].deepshrink.stop_at_2"
            :max="1"
            :min="0.05"
            :step="0.05"
          />
        </div>
      </NCard>

      <NCard :bordered="false" title="Scale">
        <div class="flex-container">
          <p class="slider-label">Scale</p>
          <NSlider
            v-model:value="target[props.tab].deepshrink.base_scale"
            :min="0.05"
            :max="1.0"
            :step="0.05"
          />
          <NInputNumber
            v-model:value="target[props.tab].deepshrink.base_scale"
            :max="1"
            :min="0.05"
            :step="0.05"
          />
        </div>
        <div class="flex-container">
          <p class="slider-label">Latent scaler</p>
          <NSelect
            v-model:value="target[props.tab].deepshrink.scaler"
            filterable
            :options="latentUpscalerOptions"
          />
        </div>
      </NCard>

      <NCard :bordered="false" title="Other">
        <div class="flex-container">
          <p class="slider-label">Early out</p>
          <NSwitch v-model:value="target[props.tab].deepshrink.early_out" />
        </div>
      </NCard>
    </div>
  </NCard>
</template>

<script setup lang="ts">
import type { ISettings } from "@/settings";
import { useSettings } from "@/store/settings";
import type { InferenceTabs } from "@/types";
import { NCard, NInputNumber, NSelect, NSlider, NSwitch } from "naive-ui";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { computed, type PropType } from "vue";

const settings = useSettings();

const latentUpscalerOptions: SelectMixedOption[] = [
  { label: "Nearest", value: "nearest" },
  { label: "Nearest exact", value: "nearest-exact" },
  { label: "Area", value: "area" },
  { label: "Bilinear", value: "bilinear" },
  { label: "Bicubic", value: "bicubic" },
  { label: "Bislerp", value: "bislerp" },
];

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
</script>
