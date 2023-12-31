export interface IAPISettings {
  websocket_sync_interval: number;
  websocket_perf_interval: number;
  enable_websocket_logging: boolean;

  use_tomesd: boolean;
  tomesd_ratio: number;
  tomesd_downsample_layers: 1 | 2 | 4 | 8;

  clip_skip: number;
  clip_quantization: "full" | "int4" | "int8";

  autocast: boolean;
  attention_processor:
    | "xformers"
    | "sdpa"
    | "cross-attention"
    | "subquadratic"
    | "multihead";
  subquadratic_size: number;
  attention_slicing: "auto" | number | "disabled";
  channels_last: boolean;
  trace_model: boolean;
  offload: "disabled" | "model" | "module";
  device: string;
  data_type: "float16" | "float32" | "bfloat16";
  deterministic_generation: boolean;
  reduced_precision: boolean;
  cudnn_benchmark: boolean;
  clear_memory_policy: "always" | "after_disconnect" | "never";
  dont_merge_latents: boolean;

  huggingface_style_parsing: boolean;

  autoloaded_textual_inversions: string[];
  autoloaded_models: string[];
  autoloaded_vae: Record<string, string>;

  save_path_template: string;
  image_extension: "webp" | "png" | "jpeg";
  image_quality: number;

  disable_grid: boolean;

  torch_compile: boolean;
  torch_compile_fullgraph: boolean;
  torch_compile_dynamic: boolean;
  torch_compile_backend: string;
  torch_compile_mode: "default" | "reduce-overhead" | "max-autotune";

  sfast_compile: boolean;
  sfast_xformers: boolean;
  sfast_triton: boolean;
  sfast_cuda_graph: boolean;

  hypertile: boolean;
  hypertile_unet_chunk: number;

  sgm_noise_multiplier: boolean;
  kdiffusers_quantization: boolean;

  xl_refiner: "joint" | "separate";

  generator: "device" | "cpu" | "philox";
  live_preview_method: "disabled" | "approximation" | "taesd";
  live_preview_delay: number;
  upcast_vae: boolean;
  vae_slicing: boolean;
  vae_tiling: boolean;
  apply_unsharp_mask: boolean;
  cfg_rescale_threshold: number | "off";

  prompt_to_prompt: boolean;
  prompt_to_prompt_model: string;
  prompt_to_prompt_device: "gpu" | "cpu";

  free_u: boolean;
  free_u_s1: number;
  free_u_s2: number;
  free_u_b1: number;
  free_u_b2: number;
}
