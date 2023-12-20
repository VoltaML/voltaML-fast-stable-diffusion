export interface HighResFixFlag {
  enabled: boolean;
  scale: number;
  mode: "latent" | "image";
  image_upscaler: string;
  latent_scale_mode:
    | "nearest"
    | "area"
    | "bilinear"
    | "bislerp"
    | "bicubic"
    | "nearest-exact";
  antialiased: boolean;
  strength: number;
  steps: number;
}

export const highresFixFlagDefault: HighResFixFlag = Object.freeze({
  enabled: false,
  scale: 2,
  mode: "image",
  image_upscaler: "RealESRGAN_x4plus_anime_6B",
  latent_scale_mode: "bislerp",
  antialiased: false,
  strength: 0.65,
  steps: 50,
});
