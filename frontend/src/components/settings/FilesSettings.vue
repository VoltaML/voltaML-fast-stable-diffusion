<template>
  <NForm>
    <NFormItem label="Template for saving outputs" label-placement="left">
      <NInput v-model:value="settings.defaultSettings.api.save_path_template" />
    </NFormItem>
    <NFormItem label="Disable generating grid image" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.disable_grid" />
    </NFormItem>
    <NFormItem label="Image extension" label-placement="left">
      <NSelect
        v-model:value="settings.defaultSettings.api.image_extension"
        :options="[
          {
            label: 'PNG',
            value: 'png',
          },
          {
            label: 'WebP',
            value: 'webp',
          },
          {
            label: 'JPEG',
            value: 'jpeg',
          },
        ]"
      />
    </NFormItem>
    <NFormItem
      label="Image quality (JPEG/WebP only)"
      label-placement="left"
      v-if="settings.defaultSettings.api.image_extension != 'png'"
    >
      <NInputNumber
        v-model:value="settings.defaultSettings.api.image_quality"
        :min="0"
        :max="100"
        :step="1"
      />
    </NFormItem>
  </NForm>
</template>

<script lang="ts" setup>
import {
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NSelect,
  NSwitch,
} from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
