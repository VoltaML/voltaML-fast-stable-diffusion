export interface SettingsInterface {
  $schema: string;
  backend: "PyTorch" | "TensorRT";
  txt2img: {
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler:
      | "Euler A"
      | "Euler"
      | "DDIM"
      | "Heun"
      | "DPM Dicsrete"
      | "DPM A"
      | "LMS"
      | "PNDM"
      | "DPMPP SDE A"
      | "DPMPP 2M";
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
    sampler: "Euler A",
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
