export enum Sampler {
  EULER_A = "Euler A",
  EULER = "Euler",
  DDIM = "DDIM",
  HEUN = "Heun",
  DPMDISCRETE = "DPM Discrete",
  DPM_A = "DPM A",
  LMS = "LMS",
  PNDM = "PNDM",
  DPMPP_SDE_A = "DPMPP SDE A",
  DPMPP_2M = "DPMPP 2M",
}

export enum KDiffusionSampler {
  EULER_A = "sample_euler_ancestral",
  EULER = "sample_euler",
  LMS = "sample_lms",
  HEUN = "sample_heun",
  DPM2 = "sample_dpm_2",
  DPM2_A = "sample_dpm_2_ancestral",
  DPMPP_2S_A = "sample_dpmpp_2s_ancestral",
  DPMPP_2M = "sample_dpmpp_2m",
  DPMPP_SDE = "sample_dpmpp_sde",
  DPM_FAST = "sample_dpm_fast",
  DPM_ADAPTIVE = "sample_dpm_adaptive",
}

export interface SettingsInterface {
  $schema: string;
  backend: "PyTorch" | "TensorRT";
  useKarrasSigmas: 1 | 0;
  model: string;
  txt2img: {
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: Sampler | KDiffusionSampler;
    prompt: string;
    negativePrompt: string;
    steps: number;
    batchCount: number;
    batchSize: number;
  };
}

const defaultSettings: SettingsInterface = {
  $schema: "./schema/ui_settings.json",
  backend: "PyTorch",
  useKarrasSigmas: 1,
  model: "none",
  txt2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: KDiffusionSampler.EULER_A,
    prompt: "",
    steps: 25,
    batchCount: 1,
    batchSize: 1,
    negativePrompt:
      "(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless",
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
