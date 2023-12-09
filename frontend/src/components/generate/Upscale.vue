<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NCard title="Upscale" class="generate-extra-card">
    <div class="flex-container">
      <p class="slider-label">Enabled</p>
      <NSwitch v-model:value="target[props.tab].upscale.enabled" />
    </div>

    <NSpace
      vertical
      class="left-container"
      v-if="target[props.tab].upscale.enabled"
    >
      <!-- Upscaler model -->
      <div class="flex-container">
        <p class="slider-label">Model</p>
        <NSelect
          v-model:value="target[props.tab].upscale.model"
          style="margin-right: 12px"
          filterable
          tag
          :options="upscalerOptionsFull"
        />
      </div>

      <!-- Scale factor -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Scale Factor</p>
          </template>
          TODO
        </NTooltip>
        <NSlider
          v-model:value="target[props.tab].upscale.upscale_factor"
          :min="1"
          :max="4"
          :step="0.1"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="target[props.tab].upscale.upscale_factor"
          size="small"
          style="min-width: 96px; width: 96px"
          :min="1"
          :max="4"
          :step="0.1"
        />
      </div>

      <!-- Tile Size -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Tile Size</p>
          </template>
          How large each tile should be. Larger tiles will use more memory. 0
          will disable tiling.
        </NTooltip>
        <NSlider
          v-model:value="target[props.tab].upscale.tile_size"
          :min="32"
          :max="2048"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="target[props.tab].upscale.tile_size"
          size="small"
          :min="32"
          :max="2048"
          style="min-width: 96px; width: 96px"
        />
      </div>

      <!-- Tile Padding -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Tile Padding</p>
          </template>
          How much should tiles overlap. Larger padding will use more memory,
          but image should not have visible seams.
        </NTooltip>
        <NSlider
          v-model:value="target[props.tab].upscale.tile_padding"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="target[props.tab].upscale.tile_padding"
          size="small"
          style="min-width: 96px; width: 96px"
        />
      </div>
    </NSpace>
  </NCard>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import type { ISettings } from "@/settings";
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
import { computed, type PropType } from "vue";
import { upscalerOptions, useSettings } from "../../store/settings";
import { useState } from "../../store/state";
import type { InferenceTabs } from "../../types";

const global = useState();
const settings = useSettings();
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

const upscalerOptionsFull = computed<SelectMixedOption[]>(() => {
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
</script>
