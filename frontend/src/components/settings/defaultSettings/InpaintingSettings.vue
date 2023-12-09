<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt" label-placement="left">
        <NInput v-model:value="settings.defaultSettings.inpainting.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt" label-placement="left">
        <NInput
          v-model:value="settings.defaultSettings.inpainting.negative_prompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.batch_count"
        />
      </NFormItem>
      <NFormItem label="Batch Size" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.batch_size"
        />
      </NFormItem>
      <NFormItem label="CFG Scale" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.cfg_scale"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Height" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.height"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Width" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.width"
          :step="8"
        />
      </NFormItem>
      <NFormItem label="Seed" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.seed"
          :min="-1"
        />
      </NFormItem>
      <NFormItem label="Steps" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.inpainting.steps"
        />
      </NFormItem>

      <SamplerPicker type="inpainting" target="defaultSettings" />
      <Upscale tab="inpainting" target="defaultSettings" />
      <HighResFix tab="inpainting" target="defaultSettings" />
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { HighResFix, SamplerPicker, Upscale } from "@/components";
import { useSettings } from "@/store/settings";
import { NCard, NForm, NFormItem, NInput, NInputNumber } from "naive-ui";

const settings = useSettings();
</script>
