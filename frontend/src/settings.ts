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
  };
}

const defaultSettings: SettingsInterface = {
  $schema: "./schema/ui_settings.json",
  backend: "PyTorch",
  txt2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: KDiffusionSampler.EULER,
    prompt: "",
    steps: 50,
    batchCount: 1,
    negativePrompt:
      "lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, jpeg artifacts, worst quality, low quality, signature, watermark, blurry, deformed, extra ears, disfigured, mutation, censored, fused legs, bad legs, bad hands, missing fingers, extra digit, fewer digits, normal quality, username, artist name",
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
