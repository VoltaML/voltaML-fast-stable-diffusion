import { ControlNetType } from "@/core/interfaces";
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { defineStore } from "pinia";
import { computed, reactive } from "vue";
import { Settings } from "../settings";

function getSchedulerOptions() {
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

export const useSettings = defineStore("settings", () => {
  const data = reactive(new Settings({}));
  const scheduler_options = computed(() => {
    return getSchedulerOptions();
  });
  const controlnet_options = computed(() => {
    return getControlNetOptions();
  });

  return { data, scheduler_options, controlnet_options };
});
