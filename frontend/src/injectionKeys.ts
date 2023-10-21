import type { InjectionKey, Ref } from "vue";
import { type ExtendedThemeOverrides } from "./types";

// Theme key
export const themeOverridesKey: InjectionKey<
  Ref<ExtendedThemeOverrides | null>
> = Symbol("theme");
