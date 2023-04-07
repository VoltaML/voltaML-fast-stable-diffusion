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

export interface SettingsInterface {
  $schema: string;
  backend: "PyTorch" | "TensorRT" | "AITemplate" | "unknown";
  model: ModelEntry | null;
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
  realesrgan: {
    image: string;
    scale_factor: number;
    model: string;
  };
  api: {
    websocket_sync_interval: number;
    websocket_perf_interval: number;
    attention_processor: "xformers" | "spda";
    attention_slicing: "auto" | number | "disabled";
    channels_last: boolean;
    vae_slicing: boolean;
    trace_model: boolean;
    offload: "module" | "model" | "disabled";
    image_preview_delay: number;
    device_id: number;
    device_type: "cpu" | "cuda" | "mps" | "directml";
    use_fp32: boolean;
  };
  aitemplate: {
    num_threads: number;
  };
  bot: {
    default_scheduler: Sampler;
    verbose: boolean;
    use_default_negative_prompt: boolean;
  };
}

export const defaultSettings: SettingsInterface = {
  $schema: "./schema/ui_data/settings.json",
  backend: "PyTorch",
  model: null,
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
  realesrgan: {
    image: "",
    scale_factor: 4,
    model: "RealESRGAN_x4plus_anime_6B",
  },
  api: {
    websocket_sync_interval: 0.02,
    websocket_perf_interval: 1,
    attention_processor: "xformers",
    attention_slicing: "disabled",
    channels_last: true,
    vae_slicing: false,
    trace_model: false,
    offload: "disabled",
    image_preview_delay: 2.0,
    device_id: 0,
    device_type: "cuda",
    use_fp32: false,
  },
  aitemplate: {
    num_threads: 8,
  },
  bot: {
    default_scheduler: Sampler.DPMSolverMultistep,
    verbose: false,
    use_default_negative_prompt: true,
  },
};

let rSettings: SettingsInterface = JSON.parse(JSON.stringify(defaultSettings));

try {
  const req = new XMLHttpRequest();
  req.open("GET", `${serverUrl}/api/settings/`, false);
  req.send();

  console.log("Recieved settings:", req.responseText);
  rSettings = JSON.parse(req.responseText);
} catch (e) {
  console.error(e);
}

console.log("Settings:", rSettings);
export const recievedSettings = rSettings;

export class Settings {
  public settings: SettingsInterface;

  constructor(settings_override: Partial<SettingsInterface>) {
    this.settings = { ...defaultSettings, ...settings_override };
  }

  public to_json(): string {
    return JSON.stringify(this.settings);
  }
}
