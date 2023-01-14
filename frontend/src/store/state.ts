import { defineStore } from "pinia";
import { reactive } from "vue";

export interface StateInterface {
  progress: number;
  generating: boolean;
  txt2img: {
    currentImage: string;
    images: string[];
  };
  current_step: number;
  total_steps: number;
}

export const useState = defineStore("state", () => {
  const state: StateInterface = reactive({
    progress: 100,
    generating: false,
    txt2img: {
      images: [],
      currentImage: "",
    },
    current_step: 0,
    total_steps: 0,
  });
  return { state };
});
