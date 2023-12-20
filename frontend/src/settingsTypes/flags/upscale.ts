export interface UpscaleFlag {
  enabled: boolean;
  upscale_factor: number;
  tile_size: number;
  tile_padding: number;
  model: string;
}

export const upscaleFlagDefault: UpscaleFlag = Object.freeze({
  enabled: false,
  upscale_factor: 4,
  tile_size: 128,
  tile_padding: 10,
  model: "RealESRGAN_x4plus_anime_6B",
});
