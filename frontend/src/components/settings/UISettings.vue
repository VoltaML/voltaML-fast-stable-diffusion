<template>
  <NForm>
    <NFormItem label="Image Preview Interval (seconds)" label-placement="left">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.live_preview_delay"
        :step="0.1"
      />
    </NFormItem>

    <NFormItem label="Image Preview Method" label-placement="left">
      <NSelect
        v-model:value="settings.defaultSettings.api.live_preview_method"
        :options="[
          { value: 'disabled', label: 'Disabled' },
          { value: 'approximation', label: 'Quick approximation (Default)' },
          { value: 'taesd', label: 'Tiny VAE' },
        ]"
      />
    </NFormItem>
    <NFormItem
      label="WebSocket Performance Monitor Interval"
      label-placement="left"
    >
      <NInputNumber
        v-model:value="settings.defaultSettings.api.websocket_perf_interval"
        :min="0.1"
        :step="0.1"
      />
    </NFormItem>
    <NFormItem label="WebSocket Sync Interval" label-placement="left">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.websocket_sync_interval"
        :min="0.001"
        :step="0.01"
      />
    </NFormItem>
  </NForm>
</template>

<script lang="ts" setup>
import { NForm, NFormItem, NInputNumber, NSelect } from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
