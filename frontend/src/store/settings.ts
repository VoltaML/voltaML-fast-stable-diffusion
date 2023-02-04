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
      label: "Euler",
      value: 5,
    },
    {
      label: "Heun",
      value: 6,
    },
    {
      label: "Euler A",
      value: 7,
    },
    {
      label: "DPM++ 2M",
      value: 8,
    },
    {
      label: "DPM++ 2S",
      value: 9,
    },
    {
      label: "DPM++ SDE",
      value: 10,
    },
    {
      label: "DPM++ 2S A Karras",
      value: 11,
    },
    {
      label: "DEIS",
      value: 12,
    },
  ];
  return scheduler_options;
}

export const useSettings = defineStore("settings", () => {
  const data = reactive(new Settings({}));
  const scheduler_options = computed(() => {
    return getSchedulerOptions();
  });

  return { data, scheduler_options };
});
