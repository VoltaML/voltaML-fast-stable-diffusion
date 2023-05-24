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

export interface ModelEntry {
  name: string;
  path: string;
  backend:
    | "TensorRT"
    | "PyTorch"
    | "AITemplate"
    | "ONNX"
    | "unknown"
    | "LoRA"
    | "Textual Inversion";
  valid: boolean;
  state: "loading" | "loaded" | "not loaded";
  loras: string[];
  textua_inversions: string[];
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
