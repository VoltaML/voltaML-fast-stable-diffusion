import { defineStore } from "pinia";
import { reactive } from "vue";

export interface StateInterface {
  progress: number;
  generating: boolean;
  txt2img: {
    currentImage: string;
    images: string[];
  };
}

export const useState = defineStore("state", () => {
  const state: StateInterface = reactive({
    progress: 100,
    generating: false,
    txt2img: {
      images: [],
      currentImage: "",
    },
  });
  return { state };
});
