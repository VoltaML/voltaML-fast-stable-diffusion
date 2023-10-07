<template>
  <NForm>
    <NFormItem label="Device" label-placement="left">
      <NSelect
        :options="availableBackends"
        v-model:value="settings.defaultSettings.api.device_type"
      />
    </NFormItem>
    <NFormItem label="Data type" label-placement="left">
      <NSelect
        :options="availableDtypes"
        v-model:value="settings.defaultSettings.api.data_type"
      />
    </NFormItem>
    <NFormItem label="Deterministic generation" label-placement="left">
      <NSwitch
        v-model:value="settings.defaultSettings.api.deterministic_generation"
      />
    </NFormItem>
    <NFormItem label="SGM Noise multiplier" label-placement="left">
      <NSwitch
        v-model:value="settings.defaultSettings.api.sgm_noise_multiplier"
      />
    </NFormItem>
    <NFormItem label="Quantization in k-samplers" label-placement="left">
      <NSwitch
        v-model:value="settings.defaultSettings.api.kdiffusers_quantization"
      />
    </NFormItem>
    <NFormItem label="Generator" label-placement="left">
      <NSelect
        v-model:value="settings.defaultSettings.api.generator"
        :options="[
          { value: 'device', label: 'On-Device' },
          { value: 'cpu', label: 'CPU' },
          { value: 'philox', label: 'CPU (device mock)' },
        ]"
      />
    </NFormItem>
    <NFormItem label="CLIP skip" label-placement="left">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.clip_skip"
        :min="1"
        :max="11"
        :step="1"
      />
    </NFormItem>
    <NFormItem
      label="CLIP quantization"
      label-placement="left"
      v-if="availableQuantizations.length != 1"
    >
      <NSelect
        :options="availableQuantizations"
        v-model:value="settings.defaultSettings.api.clip_quantization"
      />
    </NFormItem>
    <NFormItem label="ToMeSD" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.use_tomesd" />
    </NFormItem>
    <div v-if="settings.defaultSettings.api.use_tomesd">
      <NFormItem label="ToMeSD ratio" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.api.tomesd_ratio"
          :min="0"
          :max="1"
          :step="0.05"
        />
      </NFormItem>
      <NFormItem label="ToMeSD downsample layers" label-placement="left">
        <NSelect
          v-model:value="settings.defaultSettings.api.tomesd_downsample_layers"
          :options="[
            { value: '1', label: '1' },
            { value: '2', label: '2' },
            { value: '4', label: '4' },
            { value: '8', label: '8' },
          ]"
        />
      </NFormItem>
    </div>
    <NFormItem label="Huggingface-style prompting" label-placement="left">
      <NSwitch
        v-model:value="settings.defaultSettings.api.huggingface_style_parsing"
      />
    </NFormItem>
  </NForm>
</template>

<script lang="ts" setup>
import { NForm, NFormItem, NInputNumber, NSelect, NSwitch } from "naive-ui";
import { computed } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";

const settings = useSettings();
const global = useState();

const availableDtypes = computed(() => {
  if (settings.defaultSettings.api.device_type == "cpu") {
    return global.state.capabilities.supported_precisions_cpu.map((value) => {
      var description = "";
      switch (value) {
        case "float32":
          description = "32-bit float";
          break;
        case "float16":
          description = "16-bit float";
          break;
        default:
          description = "16-bit bfloat";
      }
      return { value: value, label: description };
    });
  }
  return global.state.capabilities.supported_precisions_gpu.map((value) => {
    var description = "";
    switch (value) {
      case "float32":
        description = "32-bit float";
        break;
      case "float16":
        description = "16-bit float";
        break;
      default:
        description = "16-bit bfloat";
    }
    return { value: value, label: description };
  });
});

const availableBackends = computed(() => {
  const r: SelectMixedOption[] = [];
  global.state.capabilities.supported_backends.forEach((value, key) => {
    r.push({ value: key, label: value });
  });
  return r;
});

const availableQuantizations = computed(() => {
  return [
    { value: "full", label: "Full precision" },
    ...(global.state.capabilities.supports_int8
      ? [
          { value: "int8", label: "Quantized (int8)" },
          { value: "int4", label: "Quantized (int4)" },
        ]
      : []),
  ];
});
</script>
