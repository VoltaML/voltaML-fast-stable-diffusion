import { ControlNetType } from "./core/interfaces";
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
  backend: "PyTorch" | "TensorRT" | "AITemplate";
  model: string;
  txt2img: {
    prompt: string;
    negativePrompt: string;
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: Sampler;
    steps: number;
    batchCount: number;
    batchSize: number;
  };
  img2img: {
    prompt: string;
    negativePrompt: string;
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: Sampler;
    steps: number;
    batchCount: number;
    batchSize: number;
    denoisingStrength: number;
    image: string;
  };
  imageVariations: {
    image: string;
    steps: number;
    cfgScale: number;
    seed: number;
    batchCount: number;
    batchSize: number;
    sampler: Sampler;
  };
  inpainting: {
    prompt: string;
    negativePrompt: string;
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    steps: number;
    batchCount: number;
    batchSize: number;
    sampler: Sampler;
    image: string;
    maskImage: string;
  };
  controlnet: {
    prompt: string;
    negativePrompt: string;
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    steps: number;
    batchCount: number;
    batchSize: number;
    sampler: Sampler;
    controlnet: ControlNetType;
    controlnetConditioningScale: number;
    detectionResolution: number;
    image: string;
  };
  realesrgan: {
    image: string;
    scaleFactor: number;
    model: string;
  };
  api: {
    websocketSyncInterval: number;
    websocketPerfInterval: number;
    cache_dir: string;
    optLevel: number;
    imagePreviewDelay: number;
  };
  aitemplate: {
    numThreads: number;
  };
  bot: {
    defaultScheduler: Sampler;
    verbose: boolean;
    userDefaultNegativePrompt: boolean;
  };
}

export const defaultSettings: SettingsInterface = {
  $schema: "./schema/ui_data/settings.json",
  backend: "PyTorch",
  model: "none:PyTorch",
  txt2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: Sampler.UniPCMultistep,
    prompt: "",
    steps: 25,
    batchCount: 1,
    batchSize: 1,
    negativePrompt: "",
  },
  img2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: Sampler.UniPCMultistep,
    prompt: "",
    steps: 25,
    batchCount: 1,
    batchSize: 1,
    negativePrompt: "",
    denoisingStrength: 0.6,
    image: "",
  },
  imageVariations: {
    batchCount: 1,
    batchSize: 1,
    cfgScale: 7,
    image: "",
    seed: -1,
    sampler: Sampler.UniPCMultistep,
    steps: 25,
  },
  inpainting: {
    prompt: "",
    negativePrompt: "",
    image: "",
    maskImage: "",
    width: 512,
    height: 512,
    steps: 25,
    cfgScale: 7,
    seed: -1,
    batchCount: 1,
    batchSize: 1,
    sampler: Sampler.UniPCMultistep,
  },
  controlnet: {
    prompt: "",
    image: "",
    sampler: Sampler.UniPCMultistep,
    controlnet: ControlNetType.CANNY,
    negativePrompt: "",
    width: 512,
    height: 512,
    steps: 25,
    cfgScale: 7,
    seed: -1,
    batchSize: 1,
    batchCount: 1,
    controlnetConditioningScale: 1,
    detectionResolution: 512,
  },
  realesrgan: {
    image: "",
    scaleFactor: 4,
    model: "RealESRGAN_x4plus_anime_6B",
  },
  api: {
    websocketSyncInterval: 0.02,
    websocketPerfInterval: 1,
    cache_dir: "",
    optLevel: 1,
    imagePreviewDelay: 2.0,
  },
  aitemplate: {
    numThreads: 8,
  },
  bot: {
    defaultScheduler: Sampler.UniPCMultistep,
    verbose: false,
    userDefaultNegativePrompt: true,
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
