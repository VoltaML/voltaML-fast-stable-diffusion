import type { Sampler, SigmaType } from "..";

export interface IADetailerSettings {
  enabled: boolean;

  // Inpainting
  seed: number;
  cfg_scale: number;
  steps: number;
  sampler: Sampler | string;
  self_attention_scale: number;
  sigmas: SigmaType;
  strength: number;

  // ADetailer specific
  mask_dilation: number;
  mask_blur: number;
  mask_padding: number;
  iterations: number;
  upscale: number;
}

export const defaultADetailerSettings: IADetailerSettings = {
  enabled: false,

  steps: 30,
  cfg_scale: 7,
  seed: -1,
  sampler: "dpmpp_2m",
  self_attention_scale: 0,
  sigmas: "exponential",
  strength: 0.4,

  mask_dilation: 0,
  mask_blur: 0,
  mask_padding: 0,
  iterations: 1,
  upscale: 2,
};
