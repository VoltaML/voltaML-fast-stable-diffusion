<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt">
        <NInput v-model:value="settings.defaultSettings.controlnet.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt">
        <NInput
          v-model:value="settings.defaultSettings.controlnet.negative_prompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.batch_count"
        />
      </NFormItem>
      <NFormItem label="Batch Size">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.batch_size"
        />
      </NFormItem>
      <NFormItem label="CFG Scale">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.cfg_scale"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Height">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.height"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Width">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.width"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Sampler">
        <NSelect
          :options="settings.scheduler_options"
          v-model:value="settings.defaultSettings.controlnet.sampler"
        />
      </NFormItem>
      <NFormItem label="ControlNet">
        <NSelect
          :options="settings.controlnet_options"
          v-model:value="settings.defaultSettings.controlnet.controlnet"
        />
      </NFormItem>
      <NFormItem label="Seed">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.seed"
          :min="-1"
        />
      </NFormItem>
      <NFormItem label="Steps">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.steps"
        />
      </NFormItem>
      <NFormItem label="ControlNet Conditioning Scale">
        <NInputNumber
          v-model:value="
            settings.defaultSettings.controlnet.controlnet_conditioning_scale
          "
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Detection Resolution">
        <NInputNumber
          v-model:value="
            settings.defaultSettings.controlnet.detection_resolution
          "
          :step="8"
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
