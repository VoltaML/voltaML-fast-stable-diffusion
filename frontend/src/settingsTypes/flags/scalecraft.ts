export interface IScaleCrafterFlag {
  enabled: boolean;

  base: string;
  unsafe_resolutions: boolean;
  disperse: boolean;
}

export const scaleCrafterFlagDefault: IScaleCrafterFlag = Object.freeze({
  enabled: false,

  base: "sd15",
  unsafe_resolutions: true,
  disperse: false,
});
