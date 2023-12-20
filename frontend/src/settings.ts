import { ControlNetType, type ModelEntry } from "./core/interfaces";
import { serverUrl } from "./env";
import { cloneObj } from "./functions";
import {
  Sampler,
  deepShrinkFlagDefault,
  highresFixFlagDefault,
  scaleCrafterFlagDefault,
  upscaleFlagDefault,
  type IAITemplateSettings,
  type IControlNetSettings,
  type IFlagSettings,
  type IImg2ImgSettings,
  type IInpaintingSettings,
  type ITaggerSettings,
  type ITxt2ImgSettings,
  type IUpscaleSettings,
} from "./settingsTypes";
import type { IAPISettings } from "./settingsTypes/api";
import type { IBotSettings } from "./settingsTypes/bot";
import type { IFrontendSettings } from "./settingsTypes/frontend";
import type { IONNXSettings } from "./settingsTypes/onnx";

export interface ISettings {
  $schema: string;
  backend: "PyTorch" | "AITemplate" | "ONNX" | "unknown";
  model: ModelEntry | null;
  flags: IFlagSettings;
  aitDim: {
    width: number[] | undefined;
    height: number[] | undefined;
    batch_size: number[] | undefined;
  };
  txt2img: ITxt2ImgSettings;
  img2img: IImg2ImgSettings;
  inpainting: IInpaintingSettings;
  controlnet: IControlNetSettings;
  upscale: IUpscaleSettings;
  tagger: ITaggerSettings;
  api: IAPISettings;
  aitemplate: IAITemplateSettings;
  onnx: IONNXSettings;
  bot: IBotSettings;
  frontend: IFrontendSettings;
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
