export interface IRefinerSettings {
  model: string | undefined;
  aesthetic_score: number;
  negative_aesthetic_score: number;
  steps: 50;
  strength: number;
}
