import type {
  IADetailerSettings,
  IDeepShrinkFlag,
  IHighResFixFlag,
  IScaleCrafterFlag,
  IUpscaleFlag,
  Sampler,
  SigmaType,
} from ".";

export interface IInpaintingSettings {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  seed: number;
  cfg_scale: number;
  steps: number;
  batch_count: number;
  batch_size: number;
  sampler: Sampler | string;
  image: string;
  mask_image: string;
  self_attention_scale: number;
  sigmas: SigmaType;
  highres: IHighResFixFlag;
  upscale: IUpscaleFlag;
  deepshrink: IDeepShrinkFlag;
  scalecrafter: IScaleCrafterFlag;
  adetailer: IADetailerSettings;
}
