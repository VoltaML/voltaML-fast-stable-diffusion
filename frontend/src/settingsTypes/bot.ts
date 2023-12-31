import type { Sampler } from ".";

export interface IBotSettings {
  default_scheduler: Sampler;
  verbose: boolean;
  use_default_negative_prompt: boolean;
}
