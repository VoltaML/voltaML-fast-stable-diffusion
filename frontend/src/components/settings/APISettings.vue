<template>
  <NForm>
    <h2>Saving outputs</h2>
    <NFormItem label="Template for saving outputs">
      <NInput v-model:value="settings.defaultSettings.api.save_path_template" />
    </NFormItem>

    <h2>Autoload</h2>
    <NFormItem label="Textual Inversions">
      <NSelect
        multiple
        :options="textualInversionOptions"
        v-model:value="
          settings.defaultSettings.api.autoloaded_textual_inversions
        "
      >
      </NSelect>
    </NFormItem>

    <NFormItem label="LoRAs (not functional yet)">
      <NSelect multiple :options="loraOptions"> </NSelect>
    </NFormItem>

    <h2>Timings and Queue</h2>
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

    <NFormItem label="Concurrent jobs">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.concurrent_jobs"
        :step="1"
      />
    </NFormItem>

    <h2>Optimizations</h2>

    <NFormItem label="Autocast">
      <NSwitch v-model:value="settings.defaultSettings.api.autocast" />
    </NFormItem>

    <NFormItem label="Attention Processor">
      <NSelect
        :options="[
          {
            value: 'xformers',
            label: 'xFormers (less memory hungry)',
          },
          {
            value: 'sdpa',
            label: 'SDP Attention',
          },
          {
            value: 'cross-attention',
            label: 'Cross-Attention',
          },
          {
            value: 'subquadratic',
            label: 'Sub-quadratic Attention',
          },
          {
            value: 'multihead',
            label: 'Multihead attention',
          },
        ]"
        v-model:value="settings.defaultSettings.api.attention_processor"
      >
      </NSelect>
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

    <NFormItem label="Deterministic generation">
      <NSwitch
        v-model:value="settings.defaultSettings.api.deterministic_generation"
      />
    </NFormItem>

    <NFormItem label="Reduced Precision (RTX 30xx and newer cards)">
      <NSwitch v-model:value="settings.defaultSettings.api.reduced_precision" />
    </NFormItem>

    <NFormItem
      label="CudNN Benchmark (big VRAM spikes - use on 8GB+ cards only)"
    >
      <NSwitch v-model:value="settings.defaultSettings.api.cudnn_benchmark" />
    </NFormItem>

    <NFormItem label="Clean Memory">
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

    <NFormItem label="VAE Slicing">
      <NSwitch v-model:value="settings.defaultSettings.api.vae_slicing" />
    </NFormItem>

    <NFormItem label="VAE Tiling">
      <NSwitch v-model:value="settings.defaultSettings.api.vae_tiling" />
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
            label: 'DirectML',
          },
          {
            value: 'intel',
            label: 'Intel',
          },
          {
            value: 'vulkan',
            label: 'Vulkan (Not Implemented)',
          },
          {
            value: 'iree',
            label: 'IREE (Not Implemented)',
          },
        ]"
        v-model:value="settings.defaultSettings.api.device_type"
      />
    </NFormItem>

    <NFormItem label="Device ID (GPU ID)">
      <NInputNumber v-model:value="settings.defaultSettings.api.device_id" />
    </NFormItem>

    <NFormItem label="Precision">
      <NSelect
        :options="[
          {
            value: 'float16',
            label: '16-bit float',
          },
          {
            value: 'float32',
            label: '32-bit float',
          },
          {
            value: 'bfloat16',
            label: '16-bit bfloat (CPU and Ampere+)',
          },
        ]"
        v-model:value="settings.defaultSettings.api.data_type"
      />
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
import {
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NSelect,
  NSlider,
  NSwitch,
} from "naive-ui";
import { computed } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const settings = useSettings();
const global = useState();

const textualInversions = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "Textual Inversion";
  });
});

const textualInversionOptions = computed(() => {
  return textualInversions.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
});

const loras = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "LoRA";
  });
});

const loraOptions = computed(() => {
  return loras.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
});
</script>
