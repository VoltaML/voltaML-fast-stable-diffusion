<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NCard title="Scalecrafter" class="generate-extra-card">
    <div class="flex-container">
      <div class="slider-label">
        <p>Enabled</p>
      </div>
      <NSwitch
        v-model:value="settings.defaultSettings.flags.scalecrafter.enabled"
      />
    </div>
    <NSpace
      vertical
      class="left-container"
      v-if="settings.defaultSettings.flags.scalecrafter.enabled"
    >
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Disperse</p>
          </template>
          May generate more unique images.
          <b class="highlight">
            However, this comes at the cost of increased vram usage, generally
            in the range of 3-4x.
          </b>
        </NTooltip>
        <NSwitch
          v-model:value="settings.defaultSettings.flags.scalecrafter.disperse"
        />
      </div>

      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Unsafe resolutions</p>
          </template>
          Allow generating with unique resolutions that don't have configs ready
          for them, or clamp them (really, force them) to the closest
          resolution.
        </NTooltip>
        <NSwitch
          v-model:value="
            settings.defaultSettings.flags.scalecrafter.unsafe_resolutions
          "
        />
      </div>

      <!-- Negative aesthetic score -->
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Negative Aesthetic Score</p>
          </template>
          Makes sense to keep this lower than aesthetic score.
          <b class="highlight">Generally best to keep it around 3.</b>
        </NTooltip>
        <NSlider
          v-model:value="
            settings.data.settings.flags.refiner.negative_aesthetic_score
          "
          :min="0"
          :max="10"
          :step="0.5"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="
            settings.data.settings.flags.refiner.negative_aesthetic_score
          "
          :min="0"
          :max="10"
          :step="0.25"
          size="small"
          style="min-width: 96px; width: 96px"
        />
      </div>

      <div class="flex-container">
        <p class="slider-label">Strength</p>
        <NSlider
          v-model:value="settings.data.settings.flags.refiner.strength"
          :min="0.1"
          :max="0.9"
          :step="0.05"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.data.settings.flags.refiner.strength"
          size="small"
          style="min-width: 96px; width: 96px"
          :min="0.1"
          :max="0.9"
          :step="0.05"
        />
      </div>
    </NSpace>
  </NCard>
</template>

<script setup lang="ts">
import { useSettings } from "@/store/settings";
import {
  NCard,
  NInputNumber,
  NSlider,
  NSpace,
  NSwitch,
  NTooltip,
} from "naive-ui";

const settings = useSettings();
</script>
