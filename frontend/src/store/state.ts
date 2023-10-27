import { serverUrl } from "@/env";
import { getCapabilities } from "@/helper/capabilities";
import { defineStore } from "pinia";
import { reactive, ref } from "vue";
import type { Capabilities, ModelEntry, imgData } from "../core/interfaces";
import { defaultCapabilities } from "../helper/capabilities";
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
    tab: "img2img" | "controlnet" | "inpainting";
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
  imageProcessing: {
    currentImage: string;
    images: string[];
    tab: "upscale";
  };
  extra: {
    tab: "dependencies";
  };
  tagger: {
    positivePrompt: Map<string, number>;
    negativePrompt: Map<string, number>;
    tab: "tagger";
  };
  current_step: number;
  total_steps: number;
  imageBrowser: {
    currentImage: imgData;
    currentImageByte64: string;
    currentImageMetadata: Record<string, string | number | boolean>;
  };
  perf_drawer: {
    enabled: boolean;
    gpus: GPU[];
  };
  log_drawer: {
    enabled: boolean;
    logs: string[];
  };
  models: Array<ModelEntry>;
  selected_model: ModelEntry | null;
  secrets: {
    huggingface: "missing" | "ok";
  };
  autofill: Array<string>;
  autofill_special: Array<string>;
  capabilities: Capabilities;
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
      tab: "img2img",
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
    imageProcessing: {
      images: [],
      currentImage: "",
      tab: "upscale",
    },
    extra: {
      tab: "dependencies",
    },
    tagger: {
      positivePrompt: new Map<string, number>(),
      negativePrompt: new Map<string, number>(),
      tab: "tagger",
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
      currentImageMetadata: {},
    },
    perf_drawer: {
      enabled: false,
      gpus: [],
    },
    log_drawer: {
      enabled: false,
      logs: [],
    },
    models: [],
    selected_model: ref(null),
    secrets: {
      huggingface: "ok",
    },
    autofill: [],
    autofill_special: [],
    capabilities: defaultCapabilities, // Should get replaced at runtime
  });

  async function fetchCapabilites() {
    state.capabilities = await getCapabilities();
  }

  async function fetchAutofill() {
    fetch(`${serverUrl}/api/autofill`).then(async (response) => {
      if (response.status === 200) {
        const arr: string[] = await response.json();
        state.autofill = arr;
        console.log("Autofill data successfully fetched from the server");
      } else {
        console.error("Failed to fetch autofill data");
      }
    });
  }

  return { state, fetchCapabilites, fetchAutofill };
});
