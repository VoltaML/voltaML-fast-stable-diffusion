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
}

export interface IAutoloadedLora {
  text_encoder: number;
  unet: number;
}

export interface IQuantDict {
  vae_decoder: "no-quant" | "uint8" | "int8";
  vae_encoder: "no-quant" | "uint8" | "int8";
  unet: "no-quant" | "uint8" | "int8";
  text_encoder: "no-quant" | "uint8" | "int8";
}

export interface ISettings {
  $schema: string;
  backend: "PyTorch" | "TensorRT" | "AITemplate" | "ONNX" | "unknown";
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
    width: number | undefined;
    height: number | undefined;
    batch_size: number | undefined;
  };
  txt2img: {
    prompt: string;
    negative_prompt: string;
    width: number;
    height: number;
    seed: number;
    cfg_scale: number;
    sampler: Sampler;
    steps: number;
    batch_count: number;
    batch_size: number;
    self_attention_scale: number;
  };
  img2img: {
    prompt: string;
    negative_prompt: string;
    width: number;
    height: number;
    seed: number;
    cfg_scale: number;
    sampler: Sampler;
    steps: number;
    batch_count: number;
    batch_size: number;
    denoising_strength: number;
    image: string;
    self_attention_scale: number;
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
    sampler: Sampler;
    image: string;
    mask_image: string;
    self_attention_scale: number;
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
    sampler: Sampler;
    controlnet: ControlNetType;
    controlnet_conditioning_scale: number;
    detection_resolution: number;
    image: string;
    is_preprocessed: boolean;
    save_preprocessed: boolean;
    return_preprocessed: boolean;
  };
  sd_upscale: {
    prompt: string;
    negative_prompt: string;
    seed: number;
    cfg_scale: number;
    steps: number;
    batch_count: number;
    batch_size: number;
    sampler: Sampler;
    tile_size: number;
    tile_border: number;
    original_image_slice: number;
    noise_level: number;
    image: string;
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
    concurrent_jobs: number;

    use_tomesd: boolean;
    tomesd_ratio: number;
    tomesd_downsample_layers: 1 | 2 | 4 | 8;

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
    image_preview_delay: number;
    device_id: number;
    device_type: "cpu" | "cuda" | "mps" | "directml";
    data_type: "float16" | "float32" | "bfloat16";
    deterministic_generation: boolean;
    reduced_precision: boolean;
    cudnn_benchmark: boolean;
    clear_memory_policy: "always" | "after_disconnect" | "never";

    lora_text_encoder_weight: number;
    lora_unet_weight: number;

    autoloaded_loras: Map<string, IAutoloadedLora>;
    autoloaded_textual_inversions: string[];

    save_path_template: string;
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
  };
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
  },
  sd_upscale: {
    prompt: "",
    negative_prompt: "",
    seed: -1,
    cfg_scale: 7,
    steps: 75,
    batch_count: 1,
    batch_size: 1,
    sampler: Sampler.DPMSolverMultistep,
    tile_size: 128,
    tile_border: 32,
    original_image_slice: 32,
    noise_level: 40,
    image: "",
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
    concurrent_jobs: 1,
    autocast: true,
    attention_processor: "xformers",
    subquadratic_size: 512,
    attention_slicing: "disabled",
    channels_last: true,
    vae_slicing: false,
    vae_tiling: false,
    trace_model: false,
    offload: "disabled",
    image_preview_delay: 2.0,
    device_id: 0,
    device_type: "cuda",
    data_type: "float16",
    use_tomesd: true,
    tomesd_ratio: 0.4,
    tomesd_downsample_layers: 1,
    deterministic_generation: false,
    reduced_precision: false,
    cudnn_benchmark: false,
    clear_memory_policy: "always",
    lora_text_encoder_weight: 0.5,
    lora_unet_weight: 0.5,
    autoloaded_loras: new Map(),
    autoloaded_textual_inversions: [],
    save_path_template: "{folder}/{prompt}/{id}-{index}.{extension}",
  },
  aitemplate: {
    num_threads: 8,
  },
  onnx: {
    quant_dict: {
      text_encoder: "no-quant",
      unet: "no-quant",
      vae_decoder: "no-quant",
      vae_encoder: "no-quant",
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
  },
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
