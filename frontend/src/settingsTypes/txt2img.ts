import type {
  IADetailerSettings,
  IDeepShrinkFlag,
  IHighResFixFlag,
  IScaleCrafterFlag,
  IUpscaleFlag,
  Sampler,
  SigmaType,
} from ".";

export interface ITxt2ImgSettings {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  seed: number;
  cfg_scale: number;
  sampler: Sampler | string;
  steps: number;
  batch_count: number;
  batch_size: number;
  self_attention_scale: number;
  sigmas: SigmaType;
  highres: IHighResFixFlag;
  upscale: IUpscaleFlag;
  deepshrink: IDeepShrinkFlag;
  scalecrafter: IScaleCrafterFlag;
  adetailer: IADetailerSettings;
}
