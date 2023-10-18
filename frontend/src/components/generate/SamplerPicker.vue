<template>
  <!-- Sampler -->
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
          v-for="param in Object.keys(computedSettings)"
          v-bind:key="param"
        >
          <NButton
            :type="computedSettings[param] !== null ? 'error' : 'default'"
            ghost
            :disabled="computedSettings[param] === null"
            @click="setValue(param, null)"
            style="min-width: 100px"
          >
            {{ computedSettings[param] !== null ? "Reset" : "Disabled" }}
          </NButton>
          <p style="margin-left: 12px; margin-right: 12px; white-space: nowrap">
            {{ convertToTextString(param) }}
          </p>
          <component
            :is="
              resolveComponent(
                settings.data.settings.sampler_config['ui_settings'][param],
                param
              )
            "
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

  <!-- Sigmas -->
  <div class="flex-container">
    <NTooltip style="max-width: 600px">
      <template #trigger>
        <p style="margin-right: 12px; width: 94px">Sigmas</p>
      </template>
      Changes the sigmas used in the diffusion process. Can change the quality
      of the output.
      <b class="highlight"
        >Only "Default" and "Karras" sigmas work on diffusers samplers (and
        "Karras" are only applied to KDPM samplers)</b
      >
    </NTooltip>

    <NSelect
      :options="sigmaOptions"
      v-model:value="settings.data.settings[props.type].sigmas"
      :status="sigmaValidationStatus"
    />
  </div>
</template>

<script setup lang="ts">
import { convertToTextString } from "@/functions";
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
import type { FormValidationStatus } from "naive-ui/es/form/src/interface";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import type { PropType } from "vue";
import { computed, h, ref } from "vue";
import { useSettings } from "../../store/settings";

const settings = useSettings();

const showModal = ref(false);

type SliderSettings = {
  componentType: "slider";
  min: number;
  max: number;
  step: number;
};

type SelectSettings = {
  componentType: "select";
  options: {
    label: string;
    value: string;
  };
};

type BooleanSettings = {
  componentType: "boolean";
};

type NumberInputSettings = {
  componentType: "number";
  min: number;
  max: number;
  step: number;
};

type SamplerSetting =
  | SliderSettings
  | SelectSettings
  | BooleanSettings
  | NumberInputSettings;

function getValue(param: string) {
  const val =
    settings.data.settings.sampler_config[
      settings.data.settings[props.type].sampler
    ][param];
  return val;
}

function setValue(param: string, value: any) {
  settings.data.settings.sampler_config[
    settings.data.settings[props.type].sampler
  ][param] = value;
}

function resolveComponent(settings: SamplerSetting, param: string) {
  switch (settings.componentType) {
    case "slider":
      return h(NSlider, {
        min: settings.min,
        max: settings.max,
        step: settings.step,
        value: getValue(param),
        onUpdateValue: (value: number) => setValue(param, value),
      });
    case "select":
      // @ts-ignore, some random bullshit
      return h(NSelect, {
        options: settings.options,
        value: getValue(param),
        onUpdateValue: (value: string) => setValue(param, value),
      });
    case "boolean":
      return h(NCheckbox, {
        checked: getValue(param),
        onUpdateChecked: (value: boolean) => setValue(param, value),
      });
    case "number":
      // @ts-ignore, some random bullshit
      return h(NInputNumber, {
        min: settings.min,
        max: settings.max,
        step: settings.step,
        value: getValue(param),
        onUpdateValue: (value: number) => setValue(param, value),
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

const computedSettings = computed(() => {
  return (
    settings.data.settings.sampler_config[
      settings.data.settings[props.type].sampler
    ] ?? {}
  );
});

const sigmaOptions = computed<SelectMixedOption[]>(() => {
  const karras = typeof settings.data.settings[props.type].sampler === "string";
  return [
    {
      label: "Automatic",
      value: "automatic",
    },
    {
      label: "Karras",
      value: "karras",
    },
    {
      label: "Exponential",
      value: "exponential",
      disabled: !karras,
    },
    {
      label: "Polyexponential",
      value: "polyexponential",
      disabled: !karras,
    },
    {
      label: "VP",
      value: "vp",
      disabled: !karras,
    },
  ];
});

const sigmaValidationStatus = computed<FormValidationStatus | undefined>(() => {
  if (typeof settings.data.settings[props.type].sampler !== "string") {
    if (
      !["automatic", "karras"].includes(
        settings.data.settings[props.type].sigmas
      )
    ) {
      return "error";
    } else {
      return undefined;
    }
  }
  return undefined;
});
</script>
