export interface IImg2ImgSettings {
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
  denoising_strength: number;
  image: string;
  self_attention_scale: number;
  sigmas: SigmaType;
  highres: HighResFixFlag;
  upscale: UpscaleFlag;
  deepshrink: DeepShrinkFlag;
  scalecrafter: ScaleCrafterFlag;
}
