export interface IDeepShrinkFlag {
  enabled: boolean;

  depth_1: number;
  stop_at_1: number;

  depth_2: number;
  stop_at_2: number;

  scaler: string;
  base_scale: number;
  early_out: boolean;
}

export const deepShrinkFlagDefault: IDeepShrinkFlag = Object.freeze({
  enabled: false,

  depth_1: 3,
  stop_at_1: 0.15,

  depth_2: 4,
  stop_at_2: 0.3,

  scaler: "bislerp",
  base_scale: 0.5,
  early_out: false,
});
