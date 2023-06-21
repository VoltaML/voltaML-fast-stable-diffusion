import { ControlNetType } from "@/core/interfaces";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { defineStore } from "pinia";
import { computed, reactive } from "vue";
import {
  Settings,
  defaultSettings as defaultSettingsTemplate,
  recievedSettings,
  type ISettings,
} from "../settings";

export const upscalerOptions: SelectMixedOption[] = [
  {
    label: "RealESRGAN_x4plus",
    value: "RealESRGAN_x4plus",
  },
  {
    label: "RealESRNet_x4plus",
    value: "RealESRNet_x4plus",
  },
  {
    label: "RealESRGAN_x4plus_anime_6B",
    value: "RealESRGAN_x4plus_anime_6B",
  },
  {
    label: "RealESRGAN_x2plus",
    value: "RealESRGAN_x2plus",
  },
  {
    label: "RealESR-general-x4v3",
    value: "RealESR-general-x4v3",
  },
];

export function getSchedulerOptions() {
  // Create key, value pairs for scheduler options depending on if the backend is PyTorch(KDiffusionSampler) or TensorRT(Sampler)

  const scheduler_options: SelectMixedOption[] = [
    {
      label: "DDIM",
      value: 1,
    },
    {
      label: "DDPM",
      value: 2,
    },
    {
      label: "PNDM",
      value: 3,
    },
    {
      label: "LMSD",
      value: 4,
    },
    {
      label: "EulerDiscrete",
      value: 5,
    },
    {
      label: "HeunDiscrete",
      value: 6,
    },
    {
      label: "EulerAncestralDiscrete",
      value: 7,
    },
    {
      label: "DPMSolverMultistep",
      value: 8,
    },
    {
      label: "DPMSolverSinglestep",
      value: 9,
    },
    {
      label: "KDPM2Discrete",
      value: 10,
    },
    {
      label: "KDPM2AncestralDiscrete",
      value: 11,
    },
    {
      label: "DEISMultistep",
      value: 12,
    },
    {
      label: "UniPCMultistep",
      value: 13,
    },
  ];
  return scheduler_options;
}

function getControlNetOptions() {
  const controlnet_options: SelectMixedOption[] = [
    {
      label: "Canny",
      value: ControlNetType.CANNY,
    },
    {
      label: "Depth",
      value: ControlNetType.DEPTH,
    },
    {
      label: "HED",
      value: ControlNetType.HED,
    },
    {
      label: "MLSD",
      value: ControlNetType.MLSD,
    },
    {
      label: "Normal",
      value: ControlNetType.NORMAL,
    },
    {
      label: "OpenPose",
      value: ControlNetType.OPENPOSE,
    },
    {
      label: "Scribble",
      value: ControlNetType.SCRIBBLE,
    },
    {
      label: "Segmentation",
      value: ControlNetType.SEGMENTATION,
    },
  ];
  return controlnet_options;
}

const deepcopiedSettings = JSON.parse(JSON.stringify(recievedSettings));

export const useSettings = defineStore("settings", () => {
  const data = reactive(new Settings(recievedSettings));
  const scheduler_options = computed(() => {
    return getSchedulerOptions();
  });
  const controlnet_options = computed(() => {
    return getControlNetOptions();
  });

  function resetSettings() {
    console.log("Resetting settings to default");
    Object.assign(defaultSettings, defaultSettingsTemplate);
  }

  // Deep copy default settings
  const defaultSettings: ISettings = reactive(deepcopiedSettings);

  return {
    data,
    scheduler_options,
    controlnet_options,
    defaultSettings,
    resetSettings,
  };
});
