interface IImage {
  url: string;
  nsfw: NSFWLevel;
  width: number;
  height: number;
  hash: string;
  meta: any;
}

export interface IFile {
  name: string;
  id: number;
  sizeKB: number;
  type: "Model";
  metadata: {
    fp: "fp16" | "fp32";
    size: "pruned";
    format: "SafeTensor";
  };
  pickleScanResult: string;
  pickleScanMessage: string;
  virusScanResult: string;
  scannedAt: Date;
  hashes: {
    AutoV1: string;
    AutoV2: string;
    SHA256: string;
    CRC32: string;
    BLAKE3: string;
  };
  downloadUrl: string;
  primary: boolean;
}

export interface IModelVersion {
  id: number;
  modelId: number;
  name: string;
  createdAt: string;
  updatedAt: string;
  trainedWords: string[];
  baseModel: string;
  earlyAccessTimeFrame: number;
  description: string;
  stats: {
    downloadCount: number;
    ratingCount: number;
    rating: number;
  };
  files: IFile[];
  images: IImage[];
  downloadUrl: string;
}

export interface ICivitAIModel {
  id: number;
  name: string;
  description: string;
  type: "Checkpoint";
  poi: boolean;
  nsfw: boolean;
  allowNoCredit: boolean;
  allowCommercialUse: boolean;
  allowDerivatives: boolean;
  allowDifferentLicense: boolean;
  stats: {
    downloadCount: number;
    favoriteCount: number;
    commentCount: number;
    ratingCount: number;
    rating: number;
  };
  creator: {
    username: string;
    image: string;
  };
  tags: string[];
  modelVersions: IModelVersion[];
}

export interface ICivitAIModels {
  items: ICivitAIModel[];
  metadata: {
    totalItems: number;
    currentPage: number;
    pageSize: number;
    totalPages: number;
    nextPage: string;
  };
}

export type NSFWLevel = "None" | "Soft" | "Mature" | "X";

export function nsfwIndex(nsfwLevel: NSFWLevel) {
  switch (nsfwLevel) {
    case "None":
      return 0;
    case "Soft":
      return 1;
    case "Mature":
      return 2;
    case "X":
      return 3;
  }
}
