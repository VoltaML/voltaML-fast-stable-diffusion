import type { BuiltInGlobalTheme } from "naive-ui/es/themes/interface";
import type { ComputedRef, InjectionKey, Ref } from "vue";
import type { ExtendedThemeOverrides } from "./types";

// Theme key
export const themeOverridesKey: InjectionKey<
  Ref<ExtendedThemeOverrides | null>
> = Symbol("themeOverrides");

export const themeKey: InjectionKey<ComputedRef<BuiltInGlobalTheme>> =
  Symbol("theme");
