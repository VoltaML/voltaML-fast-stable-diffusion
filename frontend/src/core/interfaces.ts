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

export enum ControlNetType {
  CANNY = "lllyasviel/sd-controlnet-canny",
  DEPTH = "lllyasviel/sd-controlnet-depth",
  HED = "lllyasviel/sd-controlnet-hed",
  MLSD = "lllyasviel/sd-controlnet-mlsd",
  NORMAL = "lllyasviel/sd-controlnet-normal",
  OPENPOSE = "lllyasviel/sd-controlnet_openpose",
  SCRIBBLE = "lllyasviel/sd-controlnet_scribble",
  SEGMENTATION = "lllyasviel/sd-controlnet_seg",
}
