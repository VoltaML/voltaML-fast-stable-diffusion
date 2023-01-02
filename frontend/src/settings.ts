export interface SettingsInterface {
  $schema: string;
  backend: "PyTorch" | "TensorRT";
  txt2img: {
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: "k_euler_a" | "ddim" | "dpm" | "lms";
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
    sampler: "k_euler_a",
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
