export interface imgData {
  id: string;
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

export enum Backends {
  "PyTorch",
  "AITemplate",
  "ONNX",
  "unknown",
  "LoRA",
  "LyCORIS",
  "VAE",
  "Textual Inversion",
  "Upscaler",
}

export type Backend = keyof typeof Backends;

export interface ModelEntry {
  name: string;
  path: string;
  backend: Backend;
  valid: boolean;
  vae: string;
  state: "loading" | "loaded" | "not loaded";
  textual_inversions: string[];
}

export interface Capabilities {
  supported_backends: string[];
  supported_precisions_gpu: string[];
  supported_precisions_cpu: string[];
  supported_torch_compile_backends: string[];
  supports_xformers: boolean;
  supports_int8: boolean;
  has_tensor_cores: boolean;
  has_tensorfloat: boolean;
}

export enum ControlNetType {
  CANNY = "lllyasviel/sd-controlnet-canny",
  DEPTH = "lllyasviel/sd-controlnet-depth",
  HED = "lllyasviel/sd-controlnet-hed",
  MLSD = "lllyasviel/sd-controlnet-mlsd",
  NORMAL = "lllyasviel/sd-controlnet-normal",
  OPENPOSE = "lllyasviel/sd-controlnet-openpose",
  SCRIBBLE = "lllyasviel/sd-controlnet-scribble",
  SEGMENTATION = "lllyasviel/sd-controlnet-seg",
}

export interface IHuggingFaceModel {
  huggingface_id: string;
  name: string;
  huggingface_url: string;
  example_image_url: string;
}
