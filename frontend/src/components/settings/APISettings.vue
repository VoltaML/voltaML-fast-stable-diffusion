<template>
  <NForm>
    <NCard title="Saving outputs" class="settings-card">
      <NFormItem label="Template for saving outputs" label-placement="left">
        <NInput
          v-model:value="settings.defaultSettings.api.save_path_template"
        />
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
    </NCard>

    <NCard title="CLIP settings" class="settings-card">
      <NFormItem label="CLIP skip" label-placement="left">
        <NInputNumber
          v-model:value="settings.defaultSettings.api.clip_skip"
          :min="1"
          :max="11"
          :step="1"
        />
      </NFormItem>
      <NFormItem label="Quantized Precision" label-placement="left">
        <NSelect
          v-model:value="settings.defaultSettings.api.clip_quantization"
          :options="availableQuantizations"
          :disabled="!global.state.capabilities.supports_int8"
        />
      </NFormItem>
    </NCard>

    <NCard title="Autoload" class="settings-card">
      <NFormItem label="Model" label-placement="left">
        <NSelect
          multiple
          filterable
          :options="autoloadModelOptions"
          v-model:value="settings.defaultSettings.api.autoloaded_models"
        >
        </NSelect>
      </NFormItem>

      <NFormItem label="Textual Inversions" label-placement="left">
        <NSelect
          multiple
          filterable
          :options="textualInversionOptions"
          v-model:value="
            settings.defaultSettings.api.autoloaded_textual_inversions
          "
        >
        </NSelect>
      </NFormItem>

      <NCard title="VAE">
        <div style="width: 100%">
          <div
            v-for="model of availableModels"
            :key="model.name"
            style="display: flex; flex-direction: row; margin-bottom: 4px"
          >
            <NText style="width: 50%">
              {{ model.name }}
            </NText>
            <NSelect
              filterable
              :options="autoloadVaeOptions"
              v-model:value="autoloadVaeValue(model.path).value"
            />
          </div>
        </div>
      </NCard>
    </NCard>

    <NCard title="Huggingface" class="settings-card">
      <NFormItem label="Huggingface-style prompting" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.api.huggingface_style_parsing"
        />
      </NFormItem>
    </NCard>

    <NCard title="Timings and Queue" class="settings-card">
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
      <NFormItem
        label="Image Preview Interval (seconds)"
        label-placement="left"
      >
        <NInputNumber
          v-model:value="settings.defaultSettings.api.image_preview_delay"
          :step="0.1"
        />
      </NFormItem>
    </NCard>

    <NCard title="Optimizations" class="settings-card">
      <NFormItem label="Autocast" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.api.autocast"
          :disabled="availableDtypes.length < 2"
        />
      </NFormItem>

      <NFormItem label="Attention Processor" label-placement="left">
        <NSelect
          :options="availableAttentions"
          v-model:value="settings.defaultSettings.api.attention_processor"
        >
        </NSelect>
      </NFormItem>

      <!-- Subquadratic attention params -->
      <div
        class="flex-container"
        v-if="
          settings.defaultSettings.api.attention_processor == 'subquadratic'
        "
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
        >
        </NSelect>
      </NFormItem>

      <NFormItem label="Channels Last" label-placement="left">
        <NSwitch v-model:value="settings.defaultSettings.api.channels_last" />
      </NFormItem>

      <NFormItem label="Deterministic generation" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.api.deterministic_generation"
        />
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

      <NFormItem label="Trace UNet" label-placement="left">
        <NSwitch v-model:value="settings.defaultSettings.api.trace_model" />
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
    </NCard>

    <NCard title="Device" class="settings-card">
      <NFormItem label="Device Type" label-placement="left">
        <NSelect
          :options="availableBackends"
          v-model:value="settings.defaultSettings.api.device_type"
        />
      </NFormItem>

      <NFormItem
        label="Device ID (GPU ID)"
        label-placement="left"
        v-if="settings.defaultSettings.api.device_type != 'cpu'"
      >
        <NInputNumber v-model:value="settings.defaultSettings.api.device_id" />
      </NFormItem>

      <NFormItem label="Precision" label-placement="left">
        <NSelect
          :options="availableDtypes"
          v-model:value="settings.defaultSettings.api.data_type"
        />
      </NFormItem>
    </NCard>

    <NCard title="Downsampling" class="settings-card">
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
            Internally splits up the generated image into a grid of this size
            and does work on them one by one. In practice, this can make
            generation up to 4x faster on <b>LARGE (1024x1024+)</b> images.
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

        <div class="flex-container">
          <NTooltip style="max-width: 600px">
            <template #trigger>
              <p class="slider-label">Hypertile VAE chunk size</p>
            </template>
            <b class="highlight"
              >PyTorch ONLY. Recommended size is a flat "128."</b
            >
            Internally splits up the generated image into a grid of this size
            and does work on them one by one. In practice, this can make VAE
            processing up to 4x faster on <b>LARGE (1024x1024+)</b> images.
          </NTooltip>

          <NSlider
            v-model:value="settings.defaultSettings.api.hypertile_vae_chunk"
            :min="128"
            :max="1024"
            :step="8"
            style="margin-right: 12px"
          />
          <NInputNumber
            v-model:value="settings.defaultSettings.api.hypertile_vae_chunk"
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
            v-model:value="
              settings.defaultSettings.api.tomesd_downsample_layers
            "
          />
        </NFormItem>
      </div>
    </NCard>

    <NCard title="Torch Compile" class="settings-card">
      <NFormItem label="Torch Compile" label-placement="left">
        <NSwitch v-model:value="settings.defaultSettings.api.torch_compile" />
      </NFormItem>

      <div v-if="settings.defaultSettings.api.torch_compile">
        <NFormItem label="Fullgraph" label-placement="left">
          <NSwitch
            v-model:value="settings.defaultSettings.api.torch_compile_fullgraph"
          />
        </NFormItem>

        <NFormItem label="Dynamic" label-placement="left">
          <NSwitch
            v-model:value="settings.defaultSettings.api.torch_compile_dynamic"
          />
        </NFormItem>

        <NFormItem label="Backend" label-placement="left">
          <NSelect
            v-model:value="settings.defaultSettings.api.torch_compile_backend"
            tag
            filterable
            :options="availableTorchCompileBackends"
          />
        </NFormItem>

        <NFormItem label="Compile Mode" label-placement="left">
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
    </NCard>
  </NForm>
</template>

<script lang="ts" setup>
import {
  NCard,
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NSelect,
  NSlider,
  NSwitch,
  NText,
  NTooltip,
} from "naive-ui";
import { computed } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

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
  return global.state.capabilities.supported_backends.map((value) => {
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

const availableModels = computed(() => {
  return global.state.models.filter((model) => {
    return (
      model.backend === "AITemplate" ||
      model.backend === "PyTorch" ||
      model.backend === "ONNX"
    );
  });
});

const availableVaes = computed(() => {
  return global.state.models.filter((model) => {
    return model.backend === "VAE";
  });
});

const autoloadModelOptions = computed(() => {
  return availableModels.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
});

const autoloadVaeOptions = computed(() => {
  const arr = availableVaes.value.map((model) => {
    return {
      value: model.path,
      label: model.name,
    };
  });
  arr.push({ value: "default", label: "Default" });

  return arr;
});

const autoloadVaeValue = (model: string) => {
  return computed({
    get: () => {
      return settings.defaultSettings.api.autoloaded_vae[model] ?? "default";
    },
    set: (value: string) => {
      if (!value || value === "default") {
        delete settings.defaultSettings.api.autoloaded_vae[model];
      } else {
        console.log("Setting", model, "to", value);
        settings.defaultSettings.api.autoloaded_vae[model] = value;
      }
    },
  });
};
</script>

<style scoped>
.settings-card {
  margin-bottom: 12px;
}
</style>
