import { ControlNetType, type ModelEntry } from "./core/interfaces";
import { serverUrl } from "./env";
import { cloneObj } from "./functions";

export enum Sampler {
  DDIM = 1,
  DDPM = 2,
  PNDM = 3,
  LMSD = 4,
  EulerDiscrete = 5,
  HeunDiscrete = 6,
  EulerAncestralDiscrete = 7,
  DPMSolverMultistep = 8,
  DPMSolverSinglestep = 9,
  KDPM2Discrete = 10,
  KDPM2AncestralDiscrete = 11,
  DEISMultistep = 12,
  UniPCMultistep = 13,
  DPMSolverSDEScheduler = 14,
}

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

const highresFixFlagDefault: HighResFixFlag = {
  enabled: false,
  scale: 2,
  mode: "image",
  image_upscaler: "RealESRGAN_x4plus_anime_6B",
  latent_scale_mode: "bislerp",
  antialiased: false,
  strength: 0.65,
  steps: 50,
};

export interface UpscaleFlag {
  enabled: boolean;
  upscale_factor: number;
  tile_size: number;
  tile_padding: number;
  model: string;
}

const upscaleFlagDefault: UpscaleFlag = {
  enabled: false,
  upscale_factor: 4,
  tile_size: 128,
  tile_padding: 10,
  model: "RealESRGAN_x4plus_anime_6B",
};

export interface DeepShrinkFlag {
  enabled: boolean;

  depth_1: number;
  stop_at_1: number;

  depth_2: number;
  stop_at_2: number;

  scaler: string;
  base_scale: number;
  early_out: boolean;
}

const deepShrinkFlagDefault: DeepShrinkFlag = {
  enabled: false,

  depth_1: 3,
  stop_at_1: 0.15,

  depth_2: 4,
  stop_at_2: 0.3,

  scaler: "bislerp",
  base_scale: 0.5,
  early_out: false,
};

export interface ScaleCrafterFlag {
  enabled: boolean;

  base: string;
  unsafe_resolutions: boolean;
  disperse: boolean;
}

const scaleCrafterFlagDefault: ScaleCrafterFlag = {
  enabled: false,

  base: "sd15",
  unsafe_resolutions: true,
  disperse: false,
};

export type SigmaType =
  | "automatic"
  | "karras"
  | "exponential"
  | "polyexponential"
  | "vp";

export interface IQuantDict {
  vae_decoder: boolean | null;
  vae_encoder: boolean | null;
  unet: boolean | null;
  text_encoder: boolean | null;
}

export interface ISettings {
  $schema: string;
  backend: "PyTorch" | "AITemplate" | "ONNX" | "unknown";
  model: ModelEntry | null;
  flags: {
    sdxl: {
      original_size: {
        width: number;
        height: number;
      };
    };
    refiner: {
      model: string | undefined;
      aesthetic_score: number;
      negative_aesthetic_score: number;
      steps: 50;
      strength: number;
    };
  };
  aitDim: {
    width: number[] | undefined;
    height: number[] | undefined;
    batch_size: number[] | undefined;
  };
  txt2img: {
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
    highres: HighResFixFlag;
    upscale: UpscaleFlag;
    deepshrink: DeepShrinkFlag;
    scalecrafter: ScaleCrafterFlag;
  };
  img2img: {
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
  };
  inpainting: {
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
    highres: HighResFixFlag;
    upscale: UpscaleFlag;
    deepshrink: DeepShrinkFlag;
    scalecrafter: ScaleCrafterFlag;
  };
  controlnet: {
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
  };
  upscale: {
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
  };
  tagger: {
    image: string;
    model: string;
    threshold: number;
  };
  api: {
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
  };
  aitemplate: {
    num_threads: number;
  };
  onnx: {
    quant_dict: IQuantDict;
    convert_to_fp16: boolean;
    simplify_unet: boolean;
  };
  bot: {
    default_scheduler: Sampler;
    verbose: boolean;
    use_default_negative_prompt: boolean;
  };
  frontend: {
    theme: string;
    enable_theme_editor: boolean;
    image_browser_columns: number;
    on_change_timer: number;
    nsfw_ok_threshold: number;
    background_image_override: string;
    disable_analytics: boolean;
  };
  sampler_config: Record<string, Record<string, any>>;
}

export const defaultSettings: ISettings = {
  $schema: "./schema/ui_data/settings.json",
  backend: "PyTorch",
  model: null,
  flags: {
    sdxl: {
      original_size: {
        width: 1024,
        height: 1024,
      },
    },
    refiner: {
      model: undefined,
      aesthetic_score: 6.0,
      negative_aesthetic_score: 2.5,
      steps: 50,
      strength: 0.3,
    },
  },
  aitDim: {
    width: undefined,
    height: undefined,
    batch_size: undefined,
  },
  txt2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfg_scale: 7,
    sampler: Sampler.DPMSolverMultistep,
    prompt: "",
    steps: 25,
    batch_count: 1,
    batch_size: 1,
    negative_prompt: "",
    self_attention_scale: 0,
    sigmas: "automatic",
    highres: cloneObj(highresFixFlagDefault),
    upscale: cloneObj(upscaleFlagDefault),
    deepshrink: cloneObj(deepShrinkFlagDefault),
    scalecrafter: cloneObj(scaleCrafterFlagDefault),
  },
  img2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfg_scale: 7,
    sampler: Sampler.DPMSolverMultistep,
    prompt: "",
    steps: 25,
    batch_count: 1,
    batch_size: 1,
    negative_prompt: "",
    denoising_strength: 0.6,
    image: "",
    self_attention_scale: 0,
    sigmas: "automatic",
    highres: cloneObj(highresFixFlagDefault),
    upscale: cloneObj(upscaleFlagDefault),
    deepshrink: cloneObj(deepShrinkFlagDefault),
    scalecrafter: cloneObj(scaleCrafterFlagDefault),
  },
  inpainting: {
    prompt: "",
    negative_prompt: "",
    image: "",
    mask_image: "",
    width: 512,
    height: 512,
    steps: 25,
    cfg_scale: 7,
    seed: -1,
    batch_count: 1,
    batch_size: 1,
    sampler: Sampler.DPMSolverMultistep,
    self_attention_scale: 0,
    sigmas: "automatic",
    highres: cloneObj(highresFixFlagDefault),
    upscale: cloneObj(upscaleFlagDefault),
    deepshrink: cloneObj(deepShrinkFlagDefault),
    scalecrafter: cloneObj(scaleCrafterFlagDefault),
  },
  controlnet: {
    prompt: "",
    image: "",
    sampler: Sampler.DPMSolverMultistep,
    controlnet: ControlNetType.CANNY,
    negative_prompt: "",
    width: 512,
    height: 512,
    steps: 25,
    cfg_scale: 7,
    seed: -1,
    batch_size: 1,
    batch_count: 1,
    controlnet_conditioning_scale: 1,
    detection_resolution: 512,
    is_preprocessed: false,
    save_preprocessed: false,
    return_preprocessed: true,
    self_attention_scale: 0.0,
    sigmas: "automatic",
    highres: cloneObj(highresFixFlagDefault),
    upscale: cloneObj(upscaleFlagDefault),
    deepshrink: cloneObj(deepShrinkFlagDefault),
    scalecrafter: cloneObj(scaleCrafterFlagDefault),
  },
  upscale: {
    image: "",
    upscale_factor: 4,
    model: "RealESRGAN_x4plus_anime_6B",
    tile_size: 128,
    tile_padding: 10,
  },
  tagger: {
    image: "",
    model: "deepdanbooru",
    threshold: 0.5,
  },
  api: {
    websocket_sync_interval: 0.02,
    websocket_perf_interval: 1,
    enable_websocket_logging: true,

    clip_skip: 1,
    clip_quantization: "full",

    autocast: true,
    attention_processor: "xformers",
    subquadratic_size: 512,
    attention_slicing: "disabled",
    channels_last: true,
    trace_model: false,
    cudnn_benchmark: false,
    offload: "disabled",
    dont_merge_latents: false,

    device: "cuda:0",
    data_type: "float16",

    use_tomesd: true,
    tomesd_ratio: 0.4,
    tomesd_downsample_layers: 1,

    deterministic_generation: false,
    reduced_precision: false,
    clear_memory_policy: "always",

    huggingface_style_parsing: false,
    autoloaded_textual_inversions: [],
    autoloaded_models: [],
    autoloaded_vae: {},

    save_path_template: "{folder}/{prompt}/{id}-{index}.{extension}",
    image_extension: "png",
    image_quality: 95,

    disable_grid: false,

    torch_compile: false,
    torch_compile_fullgraph: false,
    torch_compile_dynamic: false,
    torch_compile_backend: "inductor",
    torch_compile_mode: "default",

    sfast_compile: false,
    sfast_xformers: true,
    sfast_triton: true,
    sfast_cuda_graph: false,

    hypertile: false,
    hypertile_unet_chunk: 256,

    sgm_noise_multiplier: false,
    kdiffusers_quantization: true,

    xl_refiner: "joint",

    generator: "device",
    live_preview_method: "approximation",
    live_preview_delay: 2.0,
    upcast_vae: false,
    vae_slicing: false,
    vae_tiling: false,
    apply_unsharp_mask: false,
    cfg_rescale_threshold: 10.0,

    prompt_to_prompt: false,
    prompt_to_prompt_model: "lllyasviel/Fooocus-Expansion",
    prompt_to_prompt_device: "gpu",

    free_u: false,
    free_u_s1: 0.9,
    free_u_s2: 0.2,
    free_u_b1: 1.2,
    free_u_b2: 1.4,
  },
  aitemplate: {
    num_threads: 8,
  },
  onnx: {
    quant_dict: {
      text_encoder: null,
      unet: null,
      vae_decoder: null,
      vae_encoder: null,
    },
    convert_to_fp16: true,
    simplify_unet: false,
  },
  bot: {
    default_scheduler: Sampler.DPMSolverMultistep,
    verbose: false,
    use_default_negative_prompt: true,
  },
  frontend: {
    theme: "dark",
    enable_theme_editor: false,
    image_browser_columns: 5,
    on_change_timer: 2000,
    nsfw_ok_threshold: 0,
    background_image_override: "",
    disable_analytics: true,
  },
  sampler_config: {},
};

let rSettings: ISettings = JSON.parse(JSON.stringify(defaultSettings));

try {
  const req = new XMLHttpRequest();
  req.open("GET", `${serverUrl}/api/settings/`, false);
  req.send();

  // Merge the recieved settings with the default settings
  rSettings = { ...rSettings, ...JSON.parse(req.responseText) };
} catch (e) {
  console.error(e);
}

console.log("Settings:", rSettings);
export const recievedSettings = rSettings;

export class Settings {
  public settings: ISettings;

  constructor(settings_override: Partial<ISettings>) {
    this.settings = { ...defaultSettings, ...settings_override };
  }

  public to_json(): string {
    return JSON.stringify(this.settings);
  }
}
