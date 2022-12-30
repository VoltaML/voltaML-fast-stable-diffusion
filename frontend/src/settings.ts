interface Settings {
  txt2img: {
    width: number;
    height: number;
    seed: number;
    cfgScale: number;
    sampler: string;
    prompt: string;
    negativePrompt: string;
  };
}

export const defaultSettings: Settings = {
  txt2img: {
    width: 512,
    height: 512,
    seed: -1,
    cfgScale: 7,
    sampler: "k_euler_a",
    prompt: "",
    negativePrompt:
      "lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, jpeg artifacts, worst quality, low quality, signature, watermark, blurry, deformed, extra ears, disfigured, mutation, censored, fused legs, bad legs, bad hands, missing fingers, extra digit, fewer digits, normal quality, username, artist name",
  },
};
