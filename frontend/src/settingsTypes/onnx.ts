import type { IQuantDict } from ".";

export interface IONNXSettings {
  quant_dict: IQuantDict;
  convert_to_fp16: boolean;
  simplify_unet: boolean;
}
