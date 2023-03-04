import { defineStore } from "pinia";
import { reactive } from "vue";
import type { imgData, imgMetadata } from "../core/interfaces";
type StepProgress = "error" | "process" | "wait" | "finish";

export interface StateInterface {
  progress: number;
  generating: boolean;
  downloading: boolean;
  aitBuildStep: {
    unet: StepProgress;
    clip: StepProgress;
    vae: StepProgress;
    cleanup: StepProgress;
  };
  txt2img: {
    currentImage: string;
    images: string[];
  };
  img2img: {
    currentImage: string;
    images: string[];
  };
  inpainting: {
    currentImage: string;
    images: string[];
  };
  imageVariations: {
    currentImage: string;
    images: string[];
  };
  current_step: number;
  total_steps: number;
  imageBrowser: {
    currentImage: imgData;
    currentImageMetadata: imgMetadata;
  };
  drawer_content: string;
}

export const useState = defineStore("state", () => {
  const state: StateInterface = reactive({
    progress: 0,
    generating: false,
    downloading: false,
    aitBuildStep: {
      unet: "wait",
      clip: "wait",
      vae: "wait",
      cleanup: "wait",
    },
    txt2img: {
      images: [],
      currentImage: "",
    },
    img2img: {
      images: [],
      currentImage: "",
    },
    inpainting: {
      images: [],
      currentImage: "",
    },
    imageVariations: {
      images: [],
      currentImage: "",
    },
    current_step: 0,
    total_steps: 0,
    imageBrowser: {
      currentImage: {
        path: "",
        time: 0,
      },
      currentImageMetadata: {
        prompt: "",
        negative_prompt: "",
        width: 0,
        height: 0,
        steps: 0,
        guidance_scale: 0,
        seed: "",
        model: "",
      },
    },
    drawer_content: "Empty",
  });
  return { state };
});
