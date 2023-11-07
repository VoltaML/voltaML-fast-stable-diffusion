import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { defineStore } from "pinia";
import { computed, reactive } from "vue";
import {
  Settings,
  defaultSettings as defaultSettingsTemplate,
  recievedSettings,
  type ISettings,
} from "../settings";

export const diffusersSchedulerTuple = {
  DDIM: 1,
  DDPM: 2,
  PNDM: 3,
  LMSD: 4,
  EulerDiscrete: 5,
  HeunDiscrete: 6,
  EulerAncestralDiscrete: 7,
  DPMSolverMultistep: 8,
  DPMSolverSinglestep: 9,
  KDPM2Discrete: 10,
  KDPM2AncestralDiscrete: 11,
  DEISMultistep: 12,
  UniPCMultistep: 13,
  DPMSolverSDEScheduler: 14,
} as const;

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
  const scheduler_options: SelectMixedOption[] = [
    {
      type: "group",
      label: "k-diffusion",
      key: "K-Diffusion",
      children: [
        { label: "Euler a", value: "euler_a" },
        { label: "Euler", value: "euler" },
        { label: "LMS", value: "lms" },
        { label: "Heun", value: "heun" },
        { label: "DPM Fast", value: "dpm_fast" },
        { label: "DPM Adaptive", value: "dpm_adaptive" },
        { label: "DPM2", value: "dpm2" },
        { label: "DPM2 a", value: "dpm2_a" },
        { label: "DPM++ 2S a", value: "dpmpp_2s_a" },
        { label: "DPM++ 2M", value: "dpmpp_2m" },
        { label: "DPM++ 2M Sharp", value: "dpmpp_2m_sharp" },
        { label: "DPM++ SDE", value: "dpmpp_sde" },
        { label: "DPM++ 2M SDE", value: "dpmpp_2m_sde" },
        { label: "DPM++ 3M SDE", value: "dpmpp_3m_sde" },
        { label: "UniPC Multistep", value: "unipc_multistep" },
        { label: "Restart", value: "restart" },
      ],
    },
    {
      type: "group",
      label: "Diffusers",
      key: "diffusers",
      children: Object.keys(diffusersSchedulerTuple).map((key) => {
        return {
          label: key,
          value:
            diffusersSchedulerTuple[
              key as keyof typeof diffusersSchedulerTuple
            ],
        };
      }),
    },
  ];
  return scheduler_options;
}

function getControlNetOptions() {
  const controlnet_options: SelectMixedOption[] = [
    {
      type: "group",
      label: "ControlNet 1.1",
      key: "ControlNet 1.1",
      children: [
        {
          label: "lllyasviel/control_v11p_sd15_canny",
          value: "lllyasviel/control_v11p_sd15_canny",
        },
        {
          label: "lllyasviel/control_v11f1p_sd15_depth",
          value: "lllyasviel/control_v11f1p_sd15_depth",
        },
        {
          label: "lllyasviel/control_v11e_sd15_ip2p",
          value: "lllyasviel/control_v11e_sd15_ip2p",
        },
        {
          label: "lllyasviel/control_v11p_sd15_softedge",
          value: "lllyasviel/control_v11p_sd15_softedge",
        },
        {
          label: "lllyasviel/control_v11p_sd15_openpose",
          value: "lllyasviel/control_v11p_sd15_openpose",
        },
        {
          label: "lllyasviel/control_v11f1e_sd15_tile",
          value: "lllyasviel/control_v11f1e_sd15_tile",
        },
        {
          label: "lllyasviel/control_v11p_sd15_mlsd",
          value: "lllyasviel/control_v11p_sd15_mlsd",
        },
        {
          label: "lllyasviel/control_v11p_sd15_scribble",
          value: "lllyasviel/control_v11p_sd15_scribble",
        },
        {
          label: "lllyasviel/control_v11p_sd15_seg",
          value: "lllyasviel/control_v11p_sd15_seg",
        },
      ],
    },

    {
      type: "group",
      label: "Special",
      key: "Special",
      children: [
        {
          label: "DionTimmer/controlnet_qrcode",
          value: "DionTimmer/controlnet_qrcode",
        },
        {
          label: "CrucibleAI/ControlNetMediaPipeFace",
          value: "CrucibleAI/ControlNetMediaPipeFace",
        },
      ],
    },

    {
      type: "group",
      label: "Original",
      key: "Original",
      children: [
        {
          label: "lllyasviel/sd-controlnet-canny",
          value: "lllyasviel/sd-controlnet-canny",
        },
        {
          label: "lllyasviel/sd-controlnet-depth",
          value: "lllyasviel/sd-controlnet-depth",
        },
        {
          label: "lllyasviel/sd-controlnet-hed",
          value: "lllyasviel/sd-controlnet-hed",
        },
        {
          label: "lllyasviel/sd-controlnet-mlsd",
          value: "lllyasviel/sd-controlnet-mlsd",
        },
        {
          label: "lllyasviel/sd-controlnet-normal",
          value: "lllyasviel/sd-controlnet-normal",
        },
        {
          label: "lllyasviel/sd-controlnet-openpose",
          value: "lllyasviel/sd-controlnet-openpose",
        },
        {
          label: "lllyasviel/sd-controlnet-scribble",
          value: "lllyasviel/sd-controlnet-scribble",
        },
        {
          label: "lllyasviel/sd-controlnet-seg",
          value: "lllyasviel/sd-controlnet-seg",
        },
      ],
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
