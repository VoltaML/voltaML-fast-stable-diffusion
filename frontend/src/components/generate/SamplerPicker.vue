<template>
  <div class="flex-container">
    <NModal v-model:show="showModal" close-on-esc mask-closable>
      <NCard
        title="Sampler settings"
        style="max-width: 90vw; max-height: 90vh"
        closable
        @close="showModal = false"
      >
        <div
          class="flex-container"
          v-for="param in Object.keys(samplerSettings)"
        >
          <p style="margin-right: 12px; white-space: nowrap">{{ param }}</p>
          <component
            :is="resolveComponent(samplerSettings[param])"
            v-bind="samplerSettings[param]"
          />
        </div>
      </NCard>
    </NModal>

    <NTooltip style="max-width: 600px">
      <template #trigger>
        <p style="margin-right: 12px; width: 100px">Sampler</p>
      </template>
      The sampler is the method used to generate the image. Your result may vary
      drastically depending on the sampler you choose.
      <b class="highlight"
        >We recommend using DPMSolverMultistep for the best results .
      </b>
      <a
        target="_blank"
        href="https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
        >Learn more</a
      >
    </NTooltip>

    <NSelect
      :options="settings.scheduler_options"
      filterable
      v-model:value="settings.data.settings[props.type].sampler"
      style="flex-grow: 1"
    />

    <NButton style="margin-left: 4px" @click="showModal = true">
      <NIcon>
        <Settings />
      </NIcon>
    </NButton>
  </div>
</template>

<script setup lang="ts">
import { Settings } from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NCheckbox,
  NIcon,
  NInputNumber,
  NModal,
  NSelect,
  NSlider,
  NTooltip,
} from "naive-ui";
import type { PropType } from "vue";
import { h, ref } from "vue";
import { useSettings } from "../../store/settings";

const settings = useSettings();

const showModal = ref(false);

type SliderSettings = {
  componentType: "slider";
  min: number;
  max: number;
  step: number;
  modelValue: number;
};

type SelectSettings = {
  componentType: "select";
  options: {
    label: string;
    value: string;
  };
  modelValue: string;
};

type BooleanSettings = {
  componentType: "boolean";
  checked: boolean;
};

type NumberInputSettings = {
  componentType: "number";
  min: number;
  max: number;
  step: number;
  modelValue: number;
};

type SamplerSetting =
  | SliderSettings
  | SelectSettings
  | BooleanSettings
  | NumberInputSettings;

const samplerSettings: Record<string, SamplerSetting> = {
  eta_noise_seed_delta: {
    componentType: "number",
    min: 0,
    max: 999_999_999_999,
    step: 1,
    modelValue: 0.5,
  },
  denoiser_enable_quantization: {
    componentType: "boolean",
    checked: true,
  },
  karras_sigma_scheduler: {
    componentType: "boolean",
    checked: false,
  },
  sigma_use_old_karras_scheduler: {
    componentType: "boolean",
    checked: false,
  },
  sigma_always_discard_next_to_last: {
    componentType: "boolean",
    checked: false,
  },
  sigma_rho: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sigma_min: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sigma_max: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sampler_eta: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sampler_churn: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sampler_tmin: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sampler_tmax: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
  sampler_noise_seed_delta: {
    componentType: "slider",
    min: 0,
    max: 1,
    step: 0.1,
    modelValue: 0.5,
  },
};

function resolveComponent(settings: SamplerSetting) {
  switch (settings.componentType) {
    case "slider":
      return h(NSlider, {
        min: settings.min,
        max: settings.max,
        step: settings.step,
        modelValue: settings.modelValue,
      });
    case "select":
      return h(NSelect);
    case "boolean":
      return h(NCheckbox, {
        checked: settings.checked,
      });
    case "number":
      return h(NInputNumber, {
        min: settings.min,
        max: settings.max,
        step: settings.step,
        defaultValue: settings.modelValue,
      });
  }
}

const props = defineProps({
  type: {
    type: String as PropType<
      "txt2img" | "img2img" | "inpainting" | "controlnet"
    >,
    required: true,
  },
});
</script>
