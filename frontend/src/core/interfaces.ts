export interface imgData {
  path: string;
  time: number;
}

export interface imgMetadata {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance_scale: number;
  seed: string;
  model: string;
}

export interface ModelEntry {
  name: string;
  path: number;
  backend: "TensorRT" | "PyTorch";
  valid: boolean;
}
