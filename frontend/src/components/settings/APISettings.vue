<template>
  <NForm>
    <NFormItem label="WebSocket Performance Monitor Interval">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.websocket_perf_interval"
        :min="0.1"
        :step="0.1"
      />
    </NFormItem>
    <NFormItem label="WebSocket Sync Interval">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.websocket_sync_interval"
        :min="0.001"
        :step="0.01"
      />
    </NFormItem>
    <NFormItem label="Image Preview Interval (seconds)">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.image_preview_delay"
        :step="0.1"
      />
    </NFormItem>

    <h2>Optimizations</h2>

    <NFormItem label="Attention Processor">
      <NSelect
        :options="[
          {
            value: 'xformers',
            label: 'xFormers (less memory hungry)',
          },
          {
            value: 'spda',
            label: 'SPD Attention',
          },
        ]"
        v-model:value="settings.defaultSettings.api.attention_processor"
      >
      </NSelect>
    </NFormItem>

    <NFormItem label="Attention Slicing">
      <NSelect
        :options="[
          {
            value: 'disabled',
            label: 'None',
          },
          {
            value: 'auto',
            label: 'Auto',
          },
        ]"
        v-model:value="settings.defaultSettings.api.attention_slicing"
      >
      </NSelect>
    </NFormItem>

    <NFormItem label="Channels Last">
      <NSwitch v-model:value="settings.defaultSettings.api.channels_last" />
    </NFormItem>

    <NFormItem label="VAE Slicing">
      <NSwitch v-model:value="settings.defaultSettings.api.vae_slicing" />
    </NFormItem>

    <NFormItem label="Trace UNet">
      <NSwitch v-model:value="settings.defaultSettings.api.trace_model" />
    </NFormItem>

    <NFormItem label="Offload">
      <NSelect
        :options="[
          {
            value: 'disabled',
            label: 'Disabled',
          },
          {
            value: 'model',
            label: 'Offload the whole model to RAM when not used',
          },
          {
            value: 'module',
            label: 'Offload individual modules to RAM when not used',
          },
        ]"
        v-model:value="settings.defaultSettings.api.offload"
      >
      </NSelect>
    </NFormItem>

    <h2>Device</h2>

    <NFormItem label="Device Type">
      <NSelect
        :options="[
          {
            value: 'cpu',
            label: 'CPU',
          },
          {
            value: 'cuda',
            label: 'CUDA (NVIDIA) or ROCm (AMD)',
          },
          {
            value: 'mps',
            label: 'MPS (Apple)',
          },
          {
            value: 'directml',
            label: 'DirectML (NOT IMPLEMENTED)',
          },
        ]"
        v-model:value="settings.defaultSettings.api.device_type"
      />
    </NFormItem>

    <NFormItem label="Device ID (GPU ID)">
      <NInputNumber v-model:value="settings.defaultSettings.api.device_id" />
    </NFormItem>

    <NFormItem label="Use FP32 precision">
      <NSwitch v-model:value="settings.defaultSettings.api.use_fp32" />
    </NFormItem>

    <h2>TomeSD</h2>
    <NFormItem label="Use TomeSD">
      <NSwitch v-model:value="settings.defaultSettings.api.use_tomesd" />
    </NFormItem>

    <NFormItem label="TomeSD Ratio">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.tomesd_ratio"
        :min="0.1"
        :max="1.0"
      />
    </NFormItem>

    <NFormItem label="TomeSD Downsample layers">
      <NSelect
        :options="[
          {
            value: 1,
            label: '1',
          },
          {
            value: 2,
            label: '2',
          },
          {
            value: 4,
            label: '4',
          },
          {
            value: 8,
            label: '8',
          },
        ]"
        v-model:value="settings.defaultSettings.api.tomesd_downsample_layers"
      />
    </NFormItem>
  </NForm>
</template>

<script lang="ts" setup>
import { NForm, NFormItem, NInputNumber, NSelect, NSwitch } from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
