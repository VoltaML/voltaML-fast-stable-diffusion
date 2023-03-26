<template>
  <NCard>
    <NForm>
      <NFormItem label="WebSocket Performance Monitor Interval">
        <NInputNumber
          v-model:value="settings.defaultSettings.api.websocketPerfInterval"
          :min="0.1"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="WebSocket Sync Interval">
        <NInputNumber
          v-model:value="settings.defaultSettings.api.websocketSyncInterval"
          :min="0.001"
          :step="0.01"
        />
      </NFormItem>
      <NFormItem label="Image Preview Interval (seconds)">
        <NInputNumber
          v-model:value="settings.defaultSettings.api.imagePreviewDelay"
          :step="0.1"
        />
      </NFormItem>
      <NFormItem label="Cache Directory">
        <NInput v-model:value="settings.defaultSettings.api.cache_dir" />
      </NFormItem>

      <NFormItem label="Low VRAM mode">
        <NSwitch v-model:value="settings.defaultSettings.api.lowVRAM" />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import {
  NCard,
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NSwitch,
} from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
