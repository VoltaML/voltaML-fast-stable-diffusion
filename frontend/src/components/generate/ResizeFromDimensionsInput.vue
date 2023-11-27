<!-- eslint-disable vue/no-mutating-props -->
<template>
  <NCard title="Resize from" class="generate-extra-card">
    <div class="flex-container">
      <p class="slider-label">Enabled</p>
      <NSwitch v-model:value="global.state.txt2img.sdxl_resize" />
    </div>

    <div v-if="global.state.txt2img.sdxl_resize">
      <div class="flex-container">
        <p class="slider-label">Width</p>
        <NSlider
          v-model:value="settings.data.settings.flags.sdxl.original_size.width"
          :min="128"
          :max="2048"
          :step="1"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.sdxl.original_size.width"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="1"
        />
      </div>
      <div class="flex-container">
        <p class="slider-label">Height</p>
        <NSlider
          v-model:value="settings.data.settings.flags.sdxl.original_size.height"
          :min="128"
          :max="2048"
          :step="1"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.sdxl.original_size.height"
          size="small"
          style="min-width: 96px; width: 96px"
          :step="1"
        />
      </div>
    </div>
  </NCard>
</template>

<script setup lang="ts">
import { useSettings } from "@/store/settings";
import { useState } from "@/store/state";
import { NCard, NInputNumber, NSlider, NSwitch } from "naive-ui";

const settings = useSettings();
const global = useState();
</script>
