<template>
  <NForm>
    <NFormItem label="Device" label-placement="left">
      <NSelect
        :options="availableBackends"
        v-model:value="settings.defaultSettings.api.device"
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

    <!--TODO: check whether hypertile + tomesd can be used at once.-->
    <NFormItem label="Use HyperTile" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.hypertile" />
    </NFormItem>

    <div v-if="settings.defaultSettings.api.hypertile">
      <div class="flex-container">
        <NTooltip style="max-width: 600px">
          <template #trigger>
            <p class="slider-label">Hypertile UNet chunk size</p>
          </template>
          <b class="highlight"
            >PyTorch ONLY. Recommended sizes are 1/4th your desired resolution
            or plain "256."</b
          >
          Internally splits up the generated image into a grid of this size and
          does work on them one by one. In practice, this can make generation up
          to 4x faster on <b>LARGE (1024x1024+)</b> images.
        </NTooltip>

        <NSlider
          v-model:value="settings.defaultSettings.api.hypertile_unet_chunk"
          :min="128"
          :max="1024"
          :step="8"
          style="margin-right: 12px"
        />
        <NInputNumber
          v-model:value="settings.defaultSettings.api.hypertile_unet_chunk"
          size="small"
          style="min-width: 96px; width: 96px"
          :min="128"
          :max="1024"
          :step="1"
        />
      </div>
    </div>

    <NFormItem label="Use TomeSD" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.use_tomesd" />
    </NFormItem>
    <div v-if="settings.defaultSettings.api.use_tomesd">
      <NFormItem label="TomeSD Ratio" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.api.tomesd_ratio"
          :min="0.1"
          :max="1.0"
        />
      </NFormItem>
      <NFormItem label="TomeSD Downsample layers" label-placement="left">
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
    </div>

    <NFormItem label="Huggingface-style prompting" label-placement="left">
      <NSwitch
        v-model:value="settings.defaultSettings.api.huggingface_style_parsing"
      />
    </NFormItem>
    <NFormItem label="Prompt-to-Prompt preprocessing" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.prompt_to_prompt" />
    </NFormItem>
    <NFormItem label="Prompt-to-Prompt model" label-placement="left">
      <NSelect
        :options="[
          {
            value: 'lllyasviel/Fooocus-Expansion',
            label: 'lllyasviel/Fooocus-Expansion',
          },
          {
            value: 'daspartho/prompt-extend',
            label: 'daspartho/prompt-extend',
          },
          {
            value: 'succinctly/text2image-prompt-generator',
            label: 'succinctly/text2image-prompt-generator',
          },
          {
            value: 'Gustavosta/MagicPrompt-Stable-Diffusion',
            label: 'Gustavosta/MagicPrompt-Stable-Diffusion',
          },
          {
            value: 'Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator',
            label: 'Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator',
          },
        ]"
        v-model:value="settings.defaultSettings.api.prompt_to_prompt_model"
      />
    </NFormItem>
    <NFormItem label="Prompt-to-Prompt device" label-placement="left">
      <NSelect
        :options="[
          {
            value: 'gpu',
            label: 'On-Device',
          },
          {
            value: 'cpu',
            label: 'CPU',
          },
        ]"
        v-model:value="settings.defaultSettings.api.prompt_to_prompt_device"
      />
    </NFormItem>
  </NForm>
</template>

<script lang="ts" setup>
import {
  NForm,
  NFormItem,
  NInputNumber,
  NSelect,
  NSlider,
  NSwitch,
  NTooltip,
} from "naive-ui";
import { computed } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const settings = useSettings();
const global = useState();

const availableDtypes = computed(() => {
  if (settings.defaultSettings.api.device.includes("cpu")) {
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
  return global.state.capabilities.supported_backends.map((l) => {
    return { value: l[1], label: l[0] };
  });
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
