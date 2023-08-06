<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt">
        <NInput v-model:value="settings.defaultSettings.inpainting.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt">
        <NInput
          v-model:value="settings.defaultSettings.inpainting.negative_prompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.batch_count"
        />
      </NFormItem>
      <NFormItem label="Batch Size">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.batch_size"
        />
      </NFormItem>
      <NFormItem label="CFG Scale">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.cfg_scale"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Height">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.height"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Width">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.width"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Sampler">
        <NSelect
          :options="settings.scheduler_options"
          v-model:value="settings.defaultSettings.inpainting.sampler"
        />
      </NFormItem>
      <NFormItem label="Seed">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.seed"
          :min="-1"
        />
      </NFormItem>
      <NFormItem label="Steps">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.steps"
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
