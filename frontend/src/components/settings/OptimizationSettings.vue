<template>
  <NForm>
    <NFormItem label="Autocast" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.autocast" />
    </NFormItem>
    <NFormItem label="Attention processor" label-placement="left">
      <NSelect
        :options="availableAttentions"
        v-model:value="settings.defaultSettings.api.attention_processor"
      />
    </NFormItem>
    <!-- Subquadratic attention params -->
    <div
      class="flex-container"
      v-if="settings.defaultSettings.api.attention_processor == 'subquadratic'"
    >
      <p class="slider-label">Subquadratic chunk size (affects VRAM usage)</p>
      <NSlider
        v-model:value="settings.defaultSettings.api.subquadratic_size"
        :step="64"
        :min="64"
        :max="8192"
        style="margin-right: 12px"
      />
      <NInputNumber
        v-model:value="settings.defaultSettings.api.subquadratic_size"
        size="small"
        style="min-width: 96px; width: 96px"
        :step="64"
        :min="64"
        :max="8192"
      />
    </div>
    <NFormItem label="Compilation method" label-placement="left">
      <div flex-direction="row">
        <NButton @click="change_compilation('disabled')" :color="disableColor"
          >Disabled</NButton
        >
        <NButton @click="change_compilation('trace')" :color="traceColor"
          >Trace UNet</NButton
        >
        <NButton @click="change_compilation('compile')" :color="compileColor"
          >torch.compile</NButton
        >
      </div>
    </NFormItem>
    <div v-if="settings.defaultSettings.api.torch_compile">
      <NFormItem label="Fullgraph compile" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.api.torch_compile_fullgraph"
        />
      </NFormItem>
      <NFormItem label="Dynamic compile" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.api.torch_compile_dynamic"
        />
      </NFormItem>
      <NFormItem label="Compilation backend" label-placement="left">
        <NSelect
          :options="availableTorchCompileBackends"
          v-model:value="settings.defaultSettings.api.torch_compile_backend"
        />
      </NFormItem>
      <NFormItem label="Compilation mode" label-placement="left">
        <NSelect
          :options="[
            { value: 'default', label: 'Default' },
            { value: 'reduce-overhead', label: 'Reduce Overhead' },
            { value: 'max-autotune', label: 'Max Autotune' },
          ]"
          v-model:value="settings.defaultSettings.api.torch_compile_mode"
        />
      </NFormItem>
    </div>
    <NFormItem label="Attention Slicing" label-placement="left">
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
      />
    </NFormItem>
    <NFormItem label="Channels Last" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.channels_last" />
    </NFormItem>
    <NFormItem
      label="Reduced Precision (RTX 30xx and newer cards)"
      label-placement="left"
    >
      <NSwitch
        v-model:value="settings.defaultSettings.api.reduced_precision"
        :disabled="!global.state.capabilities.has_tensorfloat"
      />
    </NFormItem>
    <NFormItem
      label="CudNN Benchmark (big VRAM spikes - use on 8GB+ cards only)"
      label-placement="left"
    >
      <NSwitch v-model:value="settings.defaultSettings.api.cudnn_benchmark" />
    </NFormItem>
    <NFormItem label="Clean Memory" label-placement="left">
      <NSelect
        :options="[
          {
            value: 'always',
            label: 'Always',
          },
          {
            value: 'never',
            label: 'Never',
          },
          {
            value: 'after_disconnect',
            label: 'After disconnect',
          },
        ]"
        v-model:value="settings.defaultSettings.api.clear_memory_policy"
      >
      </NSelect>
    </NFormItem>
    <NFormItem label="VAE Slicing" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.vae_slicing" />
    </NFormItem>

    <NFormItem label="VAE Tiling" label-placement="left">
      <NSwitch v-model:value="settings.defaultSettings.api.vae_tiling" />
    </NFormItem>
    <NFormItem label="Offload" label-placement="left">
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
  </NForm>
</template>

<script lang="ts" setup>
import {
  NButton,
  NForm,
  NFormItem,
  NInputNumber,
  NSelect,
  NSwitch,
} from "naive-ui";
import { computed } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const settings = useSettings();
const global = useState();

const compileColor = computed(() => {
  if (settings.defaultSettings.api.torch_compile) return "#f1f1f1";
  return undefined;
});

const traceColor = computed(() => {
  if (settings.defaultSettings.api.trace_model) return "#f1f1f1";
  return undefined;
});

const disableColor = computed(() => {
  if (
    settings.defaultSettings.api.torch_compile ||
    settings.defaultSettings.api.trace_model
  )
    return undefined;
  return "#ffffff";
});

function change_compilation(a: string) {
  settings.defaultSettings.api.torch_compile = a === "compile";
  settings.defaultSettings.api.trace_model = a === "trace";
}

const availableTorchCompileBackends = computed(() => {
  return global.state.capabilities.supported_torch_compile_backends.map(
    (value) => {
      return { value: value, label: value };
    }
  );
});

const availableAttentions = computed(() => {
  return [
    ...(global.state.capabilities.supports_xformers
      ? [{ value: "xformers", label: "xFormers" }]
      : []),
    {
      value: "sdpa",
      label: "SDP Attention",
    },
    {
      value: "cross-attention",
      label: "Cross-Attention",
    },
    {
      value: "subquadratic",
      label: "Sub-quadratic Attention",
    },
    {
      value: "multihead",
      label: "Multihead attention",
    },
  ];
});
</script>
