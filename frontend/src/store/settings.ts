import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { defineStore } from "pinia";
import { computed, reactive } from "vue";
import { KDiffusionSampler, Sampler, Settings } from "../settings";

function getSchedulerOptions(backend: string) {
  // Create key, value pairs for scheduler options depending on if the backend is PyTorch(KDiffusionSampler) or TensorRT(Sampler)

  if (backend === "PyTorch") {
    const scheduler_options: SelectMixedOption[] = [
      {
        label: "Euler A",
        value: KDiffusionSampler.EULER_A,
      },
      {
        label: "Euler",
        value: KDiffusionSampler.EULER,
      },
      {
        label: "LMS",
        value: KDiffusionSampler.LMS,
      },
      {
        label: "Heun",
        value: KDiffusionSampler.HEUN,
      },
      {
        label: "DPM 2",
        value: KDiffusionSampler.DPM2,
      },
      {
        label: "DPM2 A",
        value: KDiffusionSampler.DPM2_A,
      },
      {
        label: "DPM++ 2S A",
        value: KDiffusionSampler.DPMPP_2S_A,
      },
      {
        label: "DPM++ 2M",
        value: KDiffusionSampler.DPMPP_2M,
      },
      {
        label: "DPM++ SDE",
        value: KDiffusionSampler.DPMPP_SDE,
      },
      {
        label: "DPM Fast",
        value: KDiffusionSampler.DPM_FAST,
      },
      {
        label: "DPM Adaptive",
        value: KDiffusionSampler.DPM_ADAPTIVE,
      },
    ];
    return scheduler_options;
  } else {
    const scheduler_options: SelectMixedOption[] = [
      {
        label: "Euler A",
        value: Sampler.EULER_A,
      },
      {
        label: "Euler",
        value: Sampler.EULER,
      },
      {
        label: "DDIM",
        value: Sampler.DDIM,
      },
      {
        label: "Heun",
        value: Sampler.HEUN,
      },
      {
        label: "DPM Discrete",
        value: Sampler.DPMDISCRETE,
      },
      {
        label: "DPM A",
        value: Sampler.DPM_A,
      },
      {
        label: "LMS",
        value: Sampler.LMS,
      },
      {
        label: "PNDM",
        value: Sampler.PNDM,
      },
      {
        label: "DPMPP SDE A",
        value: Sampler.DPMPP_SDE_A,
      },
      {
        label: "DPMPP 2M",
        value: Sampler.DPMPP_2M,
      },
    ];
    return scheduler_options;
  }
}

export const useSettings = defineStore("settings", () => {
  const data = reactive(new Settings({}));
  const scheduler_options = computed(() => {
    return getSchedulerOptions(data.settings.backend);
  });

  return { data, scheduler_options };
});
