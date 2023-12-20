export interface ScaleCrafterFlag {
  enabled: boolean;

  base: string;
  unsafe_resolutions: boolean;
  disperse: boolean;
}

export const scaleCrafterFlagDefault: ScaleCrafterFlag = Object.freeze({
  enabled: false,

  base: "sd15",
  unsafe_resolutions: true,
  disperse: false,
});
