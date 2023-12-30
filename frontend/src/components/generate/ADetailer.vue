<template>
  <div class="flex-container">
    <div class="slider-label">
      <p>Enabled</p>
    </div>
    <NSwitch v-model:value="target[props.tab].adetailer.enabled" />
  </div>

  <NSpace
    vertical
    class="left-container"
    v-if="target[props.tab].adetailer.enabled"
    :builtin-theme-overrides="{
      gapMedium: '0 12px',
    }"
  >
    <!-- Sampler -->
    <SamplerPicker type="inpainting" />

    <!-- Steps -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Steps</p>
        </template>
        Number of steps to take in the diffusion process. Higher values will
        result in more detailed images but will take longer to generate. There
        is also a point of diminishing returns around 100 steps.
        <b class="highlight">We recommend using 20-50 steps for most images.</b>
      </NTooltip>
      <NSlider
        v-model:value="target[props.tab].adetailer.steps"
        :min="5"
        :max="300"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="target[props.tab].adetailer.steps"
        size="small"
        style="min-width: 96px; width: 96px"
      />
    </div>

    <CFGScale tab="inpainting" target="adetailer" />
    <SAGInput tab="inpainting" target="adetailer" />

    <!-- Seed -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Seed</p>
        </template>
        Seed is a number that represents the starting canvas of your image. If
        you want to create the same image as your friend, you can use the same
        settings and seed to do so.
        <b class="highlight">For random seed use -1.</b>
      </NTooltip>
      <NInputNumber
        v-model:value="target[props.tab].adetailer.seed"
        size="small"
        :min="-1"
        :max="999_999_999_999"
        style="flex-grow: 1"
      />
    </div>

    <!-- Strength -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Strength</p>
        </template>
        How much should the masked are be changed from the original
      </NTooltip>
      <NSlider
        v-model:value="target[props.tab].adetailer.strength"
        :min="0"
        :max="1"
        :step="0.01"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="target[props.tab].adetailer.strength"
        size="small"
        style="min-width: 96px; width: 96px"
        :min="0"
        :max="1"
        :step="0.01"
      />
    </div>

    <!-- Mask Dilation -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Mask Dilation</p>
        </template>
        Expands bright pixels in the mask to cover more of the image.
      </NTooltip>
      <NInputNumber
        v-model:value="target[props.tab].adetailer.mask_dilation"
        size="small"
        :min="0"
        style="flex-grow: 1"
      />
    </div>

    <!-- Mask Blur -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Mask Blur</p>
        </template>
        Makes for a smooth transition between masked and unmasked areas.
      </NTooltip>
      <NInputNumber
        v-model:value="target[props.tab].adetailer.mask_blur"
        size="small"
        :min="0"
        style="flex-grow: 1"
      />
    </div>

    <!-- Mask Padding -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Mask Padding</p>
        </template>
        Image will be cropped to the mask size plus padding. More padding might
        mean smoother transitions but slower generation.
      </NTooltip>
      <NInputNumber
        v-model:value="target[props.tab].adetailer.mask_padding"
        size="small"
        :min="0"
        style="flex-grow: 1"
      />
    </div>

    <!-- Iterations -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Iterations</p>
        </template>
        Iterations should increase the quality of the image at the cost of time.
      </NTooltip>
      <NInputNumber
        v-model:value="target[props.tab].adetailer.iterations"
        size="small"
        :min="1"
        style="flex-grow: 1"
      />
    </div>

    <!-- Upscale -->
    <div class="flex-container">
      <NTooltip style="max-width: 600px">
        <template #trigger>
          <p class="slider-label">Upscale</p>
        </template>
        Hom much should the image be upscaled before processing. This increases
        the quality of the image at the cost of time as bigger canvas can
        usually hold more detail.
      </NTooltip>
      <NSlider
        v-model:value="target[props.tab].adetailer.upscale"
        :min="1"
        :max="4"
        :step="0.1"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="target[props.tab].adetailer.upscale"
        size="small"
        style="min-width: 96px; width: 96px"
        :min="1"
        :max="4"
        :step="0.1"
      />
    </div>
  </NSpace>
</template>

<script setup lang="ts">
import { CFGScale, SAGInput, SamplerPicker } from "@/components";
import type { ISettings } from "@/settings";
import { useSettings } from "@/store/settings";
import { NInputNumber, NSlider, NSpace, NSwitch, NTooltip } from "naive-ui";
import { computed, type PropType } from "vue";
import type { InferenceTabs } from "../../types";

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
</script>
