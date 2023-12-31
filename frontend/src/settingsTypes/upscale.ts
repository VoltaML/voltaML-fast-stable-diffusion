export interface IUpscaleSettings {
  image: string;
  upscale_factor: number;
  model:
    | "RealESRGAN_x4plus"
    | "RealESRNet_x4plus"
    | "RealESRGAN_x4plus_anime_6B"
    | "RealESRGAN_x2plus"
    | "RealESR-general-x4v3";
  tile_size: number;
  tile_padding: number;
}
