import { ControlNetType } from "./core/interfaces";

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
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: Sampler;
    prompt: string;
    negativePrompt: string;
    steps: number;
    batchCount: number;
    batchSize: number;
  };
  img2img: {
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: Sampler;
    prompt: string;
    negativePrompt: string;
    steps: number;
    batchCount: number;
    batchSize: number;
    resizeMethod: number;
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
    image: string;
    maskImage: string;
    width: number;
    height: number;
    steps: number;
    cfgScale: number;
    seed: number;
    batchCount: number;
    batchSize: number;
    sampler: Sampler;
  };
  controlnet: {
    prompt: string;
    image: string;
    sampler: Sampler;
    controlnet: ControlNetType;
    negativePrompt: string;
    width: number;
    height: number;
    steps: number;
    cfgScale: number;
    seed: number;
    batchSize: number;
    batchCount: number;
    controlnetConditioningScale: number;
    detectionResolution: number;
  };
}

export const defaultNegativePrompt =
  "(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless";

const defaultSettings: SettingsInterface = {
  $schema: "./schema/ui_settings.json",
  backend: "PyTorch",
  model: "none:PyTorch",
  txt2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: Sampler.EulerAncestralDiscrete,
    prompt: "",
    steps: 25,
    batchCount: 1,
    batchSize: 1,
    negativePrompt: defaultNegativePrompt,
  },
  img2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: Sampler.EulerAncestralDiscrete,
    prompt: "",
    steps: 25,
    batchCount: 1,
    batchSize: 1,
    negativePrompt: defaultNegativePrompt,
    denoisingStrength: 0.6,
    resizeMethod: 0,
    image: "",
  },
  imageVariations: {
    batchCount: 1,
    batchSize: 1,
    cfgScale: 7,
    image: "",
    seed: -1,
    sampler: Sampler.EulerAncestralDiscrete,
    steps: 25,
  },
  inpainting: {
    prompt: "",
    negativePrompt: defaultNegativePrompt,
    image: "",
    maskImage: "",
    width: 512,
    height: 512,
    steps: 25,
    cfgScale: 7,
    seed: -1,
    batchCount: 1,
    batchSize: 1,
    sampler: Sampler.EulerAncestralDiscrete,
  },
  controlnet: {
    prompt: "",
    image: "",
    sampler: Sampler.EulerAncestralDiscrete,
    controlnet: ControlNetType.CANNY,
    negativePrompt: defaultNegativePrompt,
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
};

export class Settings {
  public settings: SettingsInterface;

  constructor(settings_override: Partial<SettingsInterface>) {
    this.settings = { ...defaultSettings, ...settings_override };
  }

  public to_json(): string {
    return JSON.stringify(this.settings);
  }
}
