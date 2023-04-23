import { defineStore } from "pinia";
import { reactive } from "vue";
import type { imgData } from "../core/interfaces";
type StepProgress = "error" | "process" | "wait" | "finish";

export interface GPU {
  index: number;
  uuid: string;
  name: string;
  temperature: number;
  fan_speed: number;
  utilization: number;
  power_draw: number;
  power_limit: number;
  memory_used: number;
  memory_total: number;
  memory_usage: number;
}

export interface StateInterface {
  progress: number;
  generating: boolean;
  downloading: boolean;
  aitBuildStep: {
    unet: StepProgress;
    controlnet_unet: StepProgress;
    clip: StepProgress;
    vae: StepProgress;
    cleanup: StepProgress;
  };
  txt2img: {
    currentImage: string;
    highres: boolean;
    images: string[];
  };
  img2img: {
    currentImage: string;
    images: string[];
    tab: string;
  };
  inpainting: {
    currentImage: string;
    images: string[];
  };
  imageVariations: {
    currentImage: string;
    images: string[];
  };
  controlnet: {
    currentImage: string;
    images: string[];
  };
  sd_upscale: {
    currentImage: string;
    images: string[];
  };
  extra: {
    currentImage: string;
    images: string[];
    tab: string;
  };
  tagger: {
    positivePrompt: Map<string, number>;
    negativePrompt: Map<string, number>;
  };
  current_step: number;
  total_steps: number;
  imageBrowser: {
    currentImage: imgData;
    currentImageMetadata: Map<string, string>;
  };
  perf_drawer: {
    enabled: boolean;
    gpus: GPU[];
  };
}

export const useState = defineStore("state", () => {
  const state: StateInterface = reactive({
    progress: 0,
    generating: false,
    downloading: false,
    aitBuildStep: {
      unet: "wait",
      controlnet_unet: "wait",
      clip: "wait",
      vae: "wait",
      cleanup: "wait",
    },
    txt2img: {
      images: [],
      highres: false,
      currentImage: "",
    },
    img2img: {
      images: [],
      currentImage: "",
      tab: "Image to Image",
    },
    inpainting: {
      images: [],
      currentImage: "",
    },
    imageVariations: {
      images: [],
      currentImage: "",
    },
    controlnet: {
      images: [],
      currentImage: "",
    },
    sd_upscale: {
      images: [],
      currentImage: "",
    },
    extra: {
      images: [],
      currentImage: "",
      tab: "Upscale",
    },
    tagger: {
      positivePrompt: new Map<string, number>(),
      negativePrompt: new Map<string, number>(),
    },
    current_step: 0,
    total_steps: 0,
    imageBrowser: {
      currentImage: {
        path: "",
        time: 0,
      },
      currentImageMetadata: new Map(),
    },
    perf_drawer: {
      enabled: false,
      gpus: [],
    },
  });
  return { state };
});
