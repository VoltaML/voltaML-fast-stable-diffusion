import { ControlNetType, type ModelEntry } from "./core/interfaces";
import { serverUrl } from "./env";

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

export type SigmaType =
  | ""
  | "karras"
  | "automatic"
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
  extra: {
    highres: {
      scale: number;
      latent_scale_mode:
        | "nearest"
        | "area"
        | "bilinear"
        | "bislerp-original"
        | "bislerp-tortured"
        | "bicubic"
        | "nearest-exact";
      strength: number;
      steps: 50;
      antialiased: boolean;
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
    sigmas: SigmaType;
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
    vae_slicing: boolean;
    vae_tiling: boolean;
    trace_model: boolean;
    offload: "module" | "model" | "disabled";
    device: string;
    data_type: "float16" | "float32" | "bfloat16";
    deterministic_generation: boolean;
    reduced_precision: boolean;
    cudnn_benchmark: boolean;
    clear_memory_policy: "always" | "after_disconnect" | "never";

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

    sgm_noise_multiplier: boolean;
    kdiffusers_quantization: boolean;

    generator: "device" | "cpu" | "philox";
    live_preview_method: "disabled" | "approximation" | "taesd";
    live_preview_delay: number;
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
    theme: "dark" | "light";
    enable_theme_editor: boolean;
    image_browser_columns: number;
    on_change_timer: number;
    nsfw_ok_threshold: number;
  };
  sampler_config: Record<string, Record<string, any>>;
}

export const defaultSettings: ISettings = {
  $schema: "./schema/ui_data/settings.json",
  backend: "PyTorch",
  model: null,
  extra: {
    highres: {
      scale: 2,
      latent_scale_mode: "bilinear",
      strength: 0.7,
      steps: 50,
      antialiased: false,
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
    sigmas: "",
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
    sigmas: "",
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
    sigmas: "",
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
    sigmas: "",
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

    clip_skip: 1,
    clip_quantization: "full",

    autocast: true,
    attention_processor: "xformers",
    subquadratic_size: 512,
    attention_slicing: "disabled",
    channels_last: true,
    vae_slicing: false,
    vae_tiling: false,
    trace_model: false,
    cudnn_benchmark: false,
    offload: "disabled",

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

    sgm_noise_multiplier: false,
    kdiffusers_quantization: true,

    generator: "device",
    live_preview_method: "approximation",
    live_preview_delay: 2.0,
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
  },
  sampler_config: {},
};

let rSettings: ISettings = JSON.parse(JSON.stringify(defaultSettings));

try {
  const req = new XMLHttpRequest();
  req.open("GET", `${serverUrl}/api/settings/`, false);
  req.send();

  // Extra is CatchAll property, so we need to keep ours and merge the rest
  const extra = rSettings.extra;
  // Merge the recieved settings with the default settings
  rSettings = { ...rSettings, ...JSON.parse(req.responseText) };
  // Overwrite the extra property as it is store for frontend only
  Object.assign(rSettings.extra, { ...extra, ...rSettings.extra });
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
