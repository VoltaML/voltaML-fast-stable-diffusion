export interface IControlNetSettings {
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
  controlnet: ControlNetType;
  controlnet_conditioning_scale: number;
  detection_resolution: number;
  image: string;
  is_preprocessed: boolean;
  save_preprocessed: boolean;
  return_preprocessed: boolean;
  self_attention_scale: number;
  sigmas: SigmaType;
  highres: HighResFixFlag;
  upscale: UpscaleFlag;
  deepshrink: DeepShrinkFlag;
  scalecrafter: ScaleCrafterFlag;
}
