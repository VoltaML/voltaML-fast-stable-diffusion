import { defineStore } from "pinia";
import { reactive } from "vue";
import { Settings } from "../settings";

export const useSettings = defineStore("settings", () => {
  const data = reactive(new Settings({}));

  return { data };
});
