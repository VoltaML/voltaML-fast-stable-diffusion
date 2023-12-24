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

  // ADetailer specific
  mask_dilation: number;
  mask_blur: number;
  mask_padding: number;
}

export const defaultADetailerSettings: IADetailerSettings = {
  enabled: false,

  steps: 25,
  cfg_scale: 7,
  seed: -1,
  sampler: 13,
  self_attention_scale: 0,
  sigmas: "automatic",

  mask_dilation: 0,
  mask_blur: 0,
  mask_padding: 0,
};
