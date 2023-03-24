<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt">
        <NInput v-model:value="settings.defaultSettings.img2img.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt">
        <NInput
          v-model:value="settings.defaultSettings.img2img.negativePrompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.batchCount"
        />
      </NFormItem>
      <NFormItem label="Batch Size">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.batchSize"
        />
      </NFormItem>
      <NFormItem label="CFG Scale">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.cfgScale"
        />
      </NFormItem>
      <NFormItem label="Height">
        <NInputNumber v-model:value="settings.defaultSettings.img2img.height" />
      </NFormItem>
      <NFormItem label="Width">
        <NInputNumber v-model:value="settings.defaultSettings.img2img.width" />
      </NFormItem>
      <NFormItem label="Sampler">
        <NSelect
          :options="settings.scheduler_options"
          v-model:value="settings.defaultSettings.img2img.sampler"
        />
      </NFormItem>
      <NFormItem label="Seed">
        <NInputNumber v-model:value="settings.defaultSettings.img2img.seed" />
      </NFormItem>
      <NFormItem label="Steps">
        <NInputNumber v-model:value="settings.defaultSettings.img2img.steps" />
      </NFormItem>
      <NFormItem label="Denoising Strength">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.denoisingStrength"
        />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { useSettings } from "@/store/settings";
import {
  NCard,
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NSelect,
} from "naive-ui";

const settings = useSettings();
</script>
