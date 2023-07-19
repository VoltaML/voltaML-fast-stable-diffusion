<template>
  <NForm>
    <h2>Saving outputs</h2>
    <NFormItem label="Template for saving outputs">
      <NInput v-model:value="settings.defaultSettings.api.save_path_template" />
    </NFormItem>
    <NFormItem label="Disable generating grid image">
      <NSwitch v-model:value="settings.defaultSettings.api.disable_grid" />
    </NFormItem>
    <NFormItem label="Image extension">
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
    <NFormItem label="Image quality (JPEG/WebP only)">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.image_quality"
        :min="0"
        :max="100"
        :step="1"
        v-if="settings.defaultSettings.api.image_extension != 'png'"
      />
    </NFormItem>

    <h2>CLIP settings</h2>
    <NFormItem label="CLIP skip">
      <NInputNumber
        v-model:value="settings.defaultSettings.api.clip_skip"
        :min="1"
        :max="11"
        :step="1"
      />
    </NFormItem>
    <NFormItem label="Precision">
      <NSelect
        v-model:value="settings.defaultSettings.api.clip_quantization"
        :options="availableQuantizations"
        :disabled="!capabilities.supports_int8"
      />
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

    <NFormItem label="Huggingface-style prompting">
      <NSwitch
        v-model:value="settings.defaultSettings.api.huggingface_style_parsing"
      />
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

    <h2>Optimizations</h2>

    <NFormItem label="Autocast">
      <NSwitch
        v-model:value="settings.defaultSettings.api.autocast"
        :disabled="availableDtypes.length < 2"
      />
    </NFormItem>

    <NFormItem label="Attention Processor">
      <NSelect
        :options="availableAttentions"
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
      <NSwitch
        v-model:value="settings.defaultSettings.api.reduced_precision"
        :disabled="!capabilities.has_tensorfloat"
      />
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
        :options="availableBackends"
        v-model:value="settings.defaultSettings.api.device_type"
      />
    </NFormItem>

    <NFormItem
      label="Device ID (GPU ID)"
      v-if="settings.defaultSettings.api.device_type != 'cpu'"
    >
      <NInputNumber v-model:value="settings.defaultSettings.api.device_id" />
    </NFormItem>

    <NFormItem label="Precision">
      <NSelect
        :options="availableDtypes"
        v-model:value="settings.defaultSettings.api.data_type"
      />
    </NFormItem>

    <h2>TomeSD</h2>
    <NFormItem label="Use TomeSD">
      <NSwitch v-model:value="settings.defaultSettings.api.use_tomesd" />
    </NFormItem>

    <div v-if="settings.defaultSettings.api.use_tomesd">
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
    </div>

    <h2>Torch Compile</h2>
    <NFormItem label="Torch Compile">
      <NSwitch v-model:value="settings.defaultSettings.api.torch_compile" />
    </NFormItem>

    <div v-if="settings.defaultSettings.api.torch_compile">
      <NFormItem label="Fullgraph">
        <NSwitch
          v-model:value="settings.defaultSettings.api.torch_compile_fullgraph"
        />
      </NFormItem>

      <NFormItem label="Dynamic">
        <NSwitch
          v-model:value="settings.defaultSettings.api.torch_compile_dynamic"
        />
      </NFormItem>

      <NFormItem label="Backend">
        <NSelect
          v-model:value="settings.defaultSettings.api.torch_compile_backend"
          tag
          filterable
          :options="availableTorchCompileBackends"
        />
      </NFormItem>

      <NFormItem label="Compile Mode">
        <NSelect
          v-model:value="settings.defaultSettings.api.torch_compile_mode"
          :options="[
            { value: 'default', label: 'Default' },
            { value: 'reduce-overhead', label: 'Reduce Overhead' },
            { value: 'max-autotune', label: 'Max Autotune' },
          ]"
        />
      </NFormItem>
    </div>
  </NForm>
</template>

<script lang="ts" setup>
import { serverUrl } from "../../env";
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
import type { Capabilities } from "@/core/interfaces";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";

const settings = useSettings();
const global = useState();

const capabilities = computed(() => {
  try {
    const req = new XMLHttpRequest();
    req.open("GET", `${serverUrl}/api/hardware/capabilities`, false);
    req.send();

    return JSON.parse(req.responseText) as Capabilities;
  } catch (e) {
    console.error(e);
    return {
      supported_backends: ["cpu"],
      supported_precisions_cpu: ["float32"],
      supported_precisions_gpu: ["float32"],
      supported_torch_compile_backends: ["inductor"],
      has_tensorfloat: false,
      has_tensor_cores: false,
      supports_xformers: false,
      supports_int8: false,
    } as Capabilities;
  }
});

const availableDtypes = computed(() => {
  if (settings.defaultSettings.api.device_type == "cpu") {
    return capabilities.value.supported_precisions_cpu.map((value) => {
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
  return capabilities.value.supported_precisions_gpu.map((value) => {
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
  return capabilities.value.supported_backends.map((value) => {
    switch (value) {
      case "cuda":
        return { value: "cuda", label: "CUDA/ROCm" };
      case "mps":
        return { value: "mps", label: "MPS (Apple)" };
      case "xpu":
        return { value: "intel", label: "Intel" };
      case "directml":
        return { value: "directml", label: "DirectML" };
      default:
        return { value: "cpu", label: "CPU" };
    }
  });
});

const availableTorchCompileBackends = computed(() => {
  return capabilities.value.supported_torch_compile_backends.map((value) => {
    return { value: value, label: value };
  });
});

const availableAttentions = computed(() => {
  return [
    ...(capabilities.value.supports_xformers
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

const availableQuantizations = computed(() => {
  return [
    { value: "full", label: "Full precision" },
    ...(capabilities.value.supports_int8
      ? [
          { value: "int8", label: "Quantized (int8)" },
          { value: "int4", label: "Quantized (int4)" },
        ]
      : []),
  ];
});

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
</script>
