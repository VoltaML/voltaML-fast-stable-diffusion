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

      <NFormItem label="Optimization" style="">
        <NSelect
          :options="[
            {
              value: 0,
              label: '(0) Ultra speed (Traced UNet)',
            },
            {
              value: 1,
              label: '(1) Speed',
            },
            {
              value: 2,
              label: '(2) Balanced',
            },
            {
              value: 3,
              label: '(3) Multi-model VRAM efficient',
            },
            {
              value: 4,
              label: '(4) VRAM efficient',
            },
          ]"
          v-model:value="settings.defaultSettings.api.optLevel"
        >
        </NSelect>

        <a
          href="https://voltaml.github.io/voltaML-fast-stable-diffusion/experimental/optimization"
          style="
            height: 100%;
            align-self: center;
            display: flex;
            align-items: center;
            margin-left: 10px;
          "
        >
          <NIcon>
            <Help />
          </NIcon>
        </a>
      </NFormItem>

      <NFormItem label="Device ID (GPU ID)">
        <NInputNumber v-model:value="settings.defaultSettings.api.deviceID" />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { Help } from "@vicons/ionicons5";
import {
  NCard,
  NForm,
  NFormItem,
  NIcon,
  NInput,
  NInputNumber,
  NSelect,
} from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
