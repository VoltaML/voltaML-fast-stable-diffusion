import type { GlobalThemeOverrides } from "naive-ui";

export type ExtendedThemeOverrides = GlobalThemeOverrides & {
  volta: {
    base: "light" | "dark" | undefined;
    blur: string | undefined;
    backgroundImage: string | undefined;
  };
};

export type PyTorchModelBase =
  | "SD1.x"
  | "SD2.x"
  | "SDXL"
  | "Kandinsky 2.1"
  | "Kandinsky 2.2"
  | "Wuerstchen"
  | "IF";

export type PyTorchModelStage = "text_encoding" | "first_stage" | "last_stage";

export type InferenceTabs = "txt2img" | "img2img" | "inpainting" | "controlnet";
