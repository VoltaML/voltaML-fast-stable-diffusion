import { defineStore } from "pinia";
import { reactive } from "vue";
import type { ModelEntry, imgData } from "../core/interfaces";
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

export interface GenerationData {
  time_taken: number | null;
  seed: number | null;
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
  onnxBuildStep: {
    unet: StepProgress;
    clip: StepProgress;
    vae: StepProgress;
    cleanup: StepProgress;
  };
  txt2img: {
    currentImage: string;
    highres: boolean;
    images: string[];
    genData: GenerationData;
  };
  img2img: {
    currentImage: string;
    images: string[];
    tab: string;
    genData: GenerationData;
  };
  inpainting: {
    currentImage: string;
    images: string[];
    genData: GenerationData;
  };
  imageVariations: {
    currentImage: string;
    images: string[];
    genData: GenerationData;
  };
  controlnet: {
    currentImage: string;
    images: string[];
    genData: GenerationData;
  };
  sd_upscale: {
    currentImage: string;
    images: string[];
    genData: GenerationData;
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
    currentImageByte64: string;
    currentImageMetadata: Map<string, string>;
  };
  perf_drawer: {
    enabled: boolean;
    gpus: GPU[];
  };
  models: Array<ModelEntry>;
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
    onnxBuildStep: {
      unet: "wait",
      clip: "wait",
      vae: "wait",
      cleanup: "wait",
    },
    txt2img: {
      images: [],
      highres: false,
      currentImage: "",
      genData: {
        time_taken: null,
        seed: null,
      },
    },
    img2img: {
      images: [],
      currentImage: "",
      tab: "Image to Image",
      genData: {
        time_taken: null,
        seed: null,
      },
    },
    inpainting: {
      images: [],
      currentImage: "",
      genData: {
        time_taken: null,
        seed: null,
      },
    },
    imageVariations: {
      images: [],
      currentImage: "",
      genData: {
        time_taken: null,
        seed: null,
      },
    },
    controlnet: {
      images: [],
      currentImage: "",
      genData: {
        time_taken: null,
        seed: null,
      },
    },
    sd_upscale: {
      images: [],
      currentImage: "",
      genData: {
        time_taken: null,
        seed: null,
      },
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
        id: "",
        time: 0,
      },
      currentImageByte64: "",
      currentImageMetadata: new Map(),
    },
    perf_drawer: {
      enabled: false,
      gpus: [],
    },
    models: [],
  });
  return { state };
});
