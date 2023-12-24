import type { IRefinerSettings } from "./refiner";
import type { ISDXLSettings } from "./sdxl";

export * from "./adetailer";
export * from "./deepshrink";
export * from "./highres";
export * from "./refiner";
export * from "./scalecraft";
export * from "./sdxl";
export * from "./upscale";

export interface IFlagSettings {
  sdxl: ISDXLSettings;
  refiner: IRefinerSettings;
}
