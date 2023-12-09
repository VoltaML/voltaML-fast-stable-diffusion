<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt" label-placement="left">
        <NInput v-model:value="settings.defaultSettings.controlnet.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt" label-placement="left">
        <NInput
          v-model:value="settings.defaultSettings.controlnet.negative_prompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.batch_count"
        />
      </NFormItem>
      <NFormItem label="Batch Size" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.batch_size"
        />
      </NFormItem>
      <NFormItem label="CFG Scale" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.cfg_scale"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Height" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.height"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Width" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.width"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="ControlNet" label-placement="left">
        <NSelect
          :options="settings.controlnet_options"
          v-model:value="settings.defaultSettings.controlnet.controlnet"
          filterable
          tag
        />
      </NFormItem>
      <NFormItem label="Seed" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.seed"
          :min="-1"
        />
      </NFormItem>
      <NFormItem label="Is Preprocessed" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.controlnet.is_preprocessed"
        />
      </NFormItem>
      <NFormItem label="Steps" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.controlnet.steps"
        />
      </NFormItem>
      <NFormItem label="ControlNet Conditioning Scale" label-placement="left">
        <NInputNumber
          v-model:value="
            settings.defaultSettings.controlnet.controlnet_conditioning_scale
          "
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Detection Resolution" label-placement="left">
        <NInputNumber
          v-model:value="
            settings.defaultSettings.controlnet.detection_resolution
          "
          :step="8"
        />
      </NFormItem>

      <SamplerPicker type="controlnet" target="defaultSettings" />
      <Upscale tab="controlnet" target="defaultSettings" />
      <HighResFix tab="controlnet" target="defaultSettings" />
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { HighResFix, SamplerPicker, Upscale } from "@/components";
import { useSettings } from "@/store/settings";
import {
  NCard,
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NSelect,
  NSwitch,
} from "naive-ui";

const settings = useSettings();
</script>
