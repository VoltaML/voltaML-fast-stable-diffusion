<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt" label-placement="left">
        <NInput v-model:value="settings.defaultSettings.img2img.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt" label-placement="left">
        <NInput
          v-model:value="settings.defaultSettings.img2img.negative_prompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.batch_count"
        />
      </NFormItem>
      <NFormItem label="Batch Size" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.batch_size"
        />
      </NFormItem>
      <NFormItem label="CFG Scale" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.cfg_scale"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Height" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.height"
          :step="1"
        />
      </NFormItem>
      <NFormItem label="Width" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.width"
          :step="1"
        />
      </NFormItem>
      <NFormItem label="Sampler" label-placement="left">
        <NSelect
          :options="settings.scheduler_options"
          v-model:value="settings.defaultSettings.img2img.sampler"
        />
      </NFormItem>
      <NFormItem label="Seed" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.seed"
          :min="-1"
        />
      </NFormItem>
      <NFormItem label="Steps" label-placement="left">
        <NInputNumber v-model:value="settings.defaultSettings.img2img.steps" />
      </NFormItem>
      <NFormItem label="Denoising Strength" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.img2img.denoising_strength"
          :step="0.1"
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
