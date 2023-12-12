<template>
  <NCard>
    <NForm>
      <NFormItem label="Prompt" label-placement="left">
        <NInput v-model:value="settings.defaultSettings.txt2img.prompt" />
      </NFormItem>
      <NFormItem label="Negative Prompt" label-placement="left">
        <NInput
          v-model:value="settings.defaultSettings.txt2img.negative_prompt"
        />
      </NFormItem>
      <NFormItem label="Batch Count" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.txt2img.batch_count"
        />
      </NFormItem>
      <NFormItem label="Batch Size" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.txt2img.batch_size"
        />
      </NFormItem>
      <NFormItem label="CFG Scale" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.txt2img.cfg_scale"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Height" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.txt2img.height"
          :step="1"
        />
      </NFormItem>
      <NFormItem label="Width" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.txt2img.width"
          :step="1"
        />
      </NFormItem>
      <NFormItem label="Seed" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.txt2img.seed"
          :min="-1"
        />
      </NFormItem>
      <NFormItem label="Steps" label-placement="left">
        <NInputNumber v-model:value="settings.defaultSettings.txt2img.steps" />
      </NFormItem>

      <SamplerPicker type="txt2img" target="defaultSettings" />
      <Upscale tab="txt2img" target="defaultSettings" />
      <HighResFixTabs tab="txt2img" target="defaultSettings" />
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { HighResFixTabs, SamplerPicker, Upscale } from "@/components";
import { NCard, NForm, NFormItem, NInput, NInputNumber } from "naive-ui";
import { useSettings } from "../../../store/settings";

const settings = useSettings();
</script>
