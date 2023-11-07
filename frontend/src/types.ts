import type { GlobalThemeOverrides } from "naive-ui";

export type ExtendedThemeOverrides = GlobalThemeOverrides & {
  volta: {
    base: "light" | "dark" | undefined;
    blur: string | undefined;
    backgroundImage: string | undefined;
  };
};

export type InferenceTabs = "txt2img" | "img2img" | "inpainting" | "controlnet";
