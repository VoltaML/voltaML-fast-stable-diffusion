export interface Model {
  huggingface_id: string;
  name: string;
  tags: string[];
  huggingface_url: string;
  example_image_url: string;
}

export const tagColor: {
  [key: string]:
    | "default"
    | "success"
    | "error"
    | "warning"
    | "primary"
    | "info";
} = {
  anime: "warning",
  stylized: "info",
  original: "primary",
  highQuality: "error",
  dreambooth: "default",
};

export const modelData: Model[] = [
  {
    name: "Anything V3",
    huggingface_id: "Linaqruf/anything-v3.0",
    huggingface_url: "https://huggingface.co/Linaqruf/anything-v3.0",
    tags: ["anime"],
    example_image_url:
      "https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/1girl.png",
  },
  {
    name: "ACertainThing",
    huggingface_id: "JosephusCheung/ACertainThing",
    huggingface_url: "https://huggingface.co/JosephusCheung/ACertainThing",
    tags: ["anime", "dreambooth"],
    example_image_url:
      "https://huggingface.co/JosephusCheung/ACertainThing/resolve/main/samples/acth-sample-1girl.png",
  },
  {
    name: "SD-Kurzgesagt",
    huggingface_id: "questcoast/SD-Kurzgesagt-style-finetune",
    huggingface_url:
      "https://huggingface.co/questcoast/SD-Kurzgesagt-style-finetune",
    tags: ["stylized", "dreambooth"],
    example_image_url:
      "https://huggingface.co/questcoast/SD-Kurzgesagt-style-finetune/resolve/main/samples-2.jpg",
  },
  {
    name: "Stable Diffusion v1.5",
    huggingface_id: "runwayml/stable-diffusion-v1-5",
    huggingface_url: "https://huggingface.co/runwayml/stable-diffusion-v1-5",
    tags: ["original"],
    example_image_url: "",
  },
  {
    name: "Vintedois Diffusion",
    huggingface_id: "22h/vintedois-diffusion-v0-1",
    huggingface_url: "https://huggingface.co/22h/vintedois-diffusion-v0-1",
    tags: ["highQuality"],
    example_image_url:
      "https://huggingface.co/22h/vintedois-diffusion-v0-1/resolve/main/44-euler-a-kneeling%20cat%20knight%2C%20portrait%2C%20finely%20detailed%20armor%2C%20intricate%20design%2C%20silver%2C%20silk%2C%20cinematic%20lighting%2C%204k.png",
  },
  {
    name: "Dreamlike diffusion",
    huggingface_id: "dreamlike-art/dreamlike-diffusion-1.0",
    huggingface_url:
      "https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0",
    tags: ["highQuality"],
    example_image_url:
      "https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/resolve/main/preview.jpg",
  },
  {
    name: "Stable Diffusion v1.4",
    huggingface_id: "CompVis/stable-diffusion-v1-4",
    huggingface_url: "https://huggingface.co/CompVis/stable-diffusion-v1-4",
    tags: ["original"],
    example_image_url: "",
  },
  {
    name: "OpenJourney",
    huggingface_id: "prompthero/openjourney",
    huggingface_url: "https://huggingface.co/prompthero/openjourney",
    tags: ["highQuality"],
    example_image_url:
      "https://s3.amazonaws.com/moonup/production/uploads/1667904587642-63265d019f9d19bfd4f45031.png",
  },
  {
    name: "RedShift Diffusion",
    huggingface_id: "nitrosocke/redshift-diffusion",
    huggingface_url: "https://huggingface.co/nitrosocke/redshift-diffusion",
    tags: ["highQuality"],
    example_image_url:
      "https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/images/redshift-diffusion-samples-01s.jpg",
  },
];
