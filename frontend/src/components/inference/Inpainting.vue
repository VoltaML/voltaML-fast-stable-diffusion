<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <div style="margin: 0 12px">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <NCard title="Input image">
          <div class="image-container" ref="imageContainer">
            <VueDrawingCanvas
              :width="width"
              :height="height"
              :backgroundImage="preview"
              :lineWidth="strokeWidth"
              strokeType="dash"
              lineCap="round"
              lineJoin="round"
              :fillShape="false"
              :eraser="eraser"
              color="black"
              ref="canvas"
              saveAs="png"
              canvas-id="VueDrawingCanvas1"
            />
            <VueDrawingCanvas
              v-model:image="settings.data.settings.inpainting.mask_image"
              :width="width"
              :height="height"
              ref="maskCanvas"
              saveAs="png"
              style="display: none"
              canvas-id="VueDrawingCanvas2"
            />
          </div>

          <NSpace
            inline
            justify="space-between"
            align="center"
            style="width: 100%; margin-top: 12px"
          >
            <div style="display: inline-flex; align-items: center">
              <NButton class="utility-button" @click="undo">
                <NIcon>
                  <ArrowUndoSharp />
                </NIcon>
              </NButton>
              <NButton class="utility-button" @click="redo">
                <NIcon>
                  <ArrowRedoSharp />
                </NIcon>
              </NButton>
              <NButton class="utility-button" @click="toggleEraser">
                <NIcon v-if="eraser">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                    fill="currentColor"
                    class="bi bi-eraser"
                    viewBox="0 0 16 16"
                  >
                    <path
                      d="M8.086 2.207a2 2 0 0 1 2.828 0l3.879 3.879a2 2 0 0 1 0 2.828l-5.5 5.5A2 2 0 0 1 7.879 15H5.12a2 2 0 0 1-1.414-.586l-2.5-2.5a2 2 0 0 1 0-2.828l6.879-6.879zm2.121.707a1 1 0 0 0-1.414 0L4.16 7.547l5.293 5.293 4.633-4.633a1 1 0 0 0 0-1.414l-3.879-3.879zM8.746 13.547 3.453 8.254 1.914 9.793a1 1 0 0 0 0 1.414l2.5 2.5a1 1 0 0 0 .707.293H7.88a1 1 0 0 0 .707-.293l.16-.16z"
                    />
                  </svg>
                </NIcon>
                <NIcon v-else>
                  <BrushSharp />
                </NIcon>
              </NButton>
              <NButton class="utility-button" @click="clearCanvas">
                <NIcon>
                  <TrashBinSharp />
                </NIcon>
              </NButton>
              <NSlider
                v-model:value="strokeWidth"
                :min="1"
                :max="50"
                :step="1"
                style="width: 100px; margin: 0 8px"
              />
              <p>{{ width }}x{{ height }}</p>
            </div>
            <label for="file-upload">
              <span class="file-upload">Select image</span>
            </label>
          </NSpace>
          <input
            type="file"
            accept="image/*"
            @change="previewImage"
            id="file-upload"
            class="hidden-input"
          />
        </NCard>

        <NCard title="Settings" style="margin-top: 12px; margin-bottom: 12px">
          <NSpace vertical class="left-container">
            <!-- Prompt -->
            <Prompt tab="inpainting" />

            <!-- Sampler -->
            <SamplerPicker type="inpainting" />

            <!-- Dimensions -->
            <div class="flex-container">
              <p class="slider-label">Width</p>
              <NSlider
                v-model:value="settings.data.settings.inpainting.width"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.inpainting.width"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="128"
                :max="2048"
              />
            </div>
            <div class="flex-container">
              <p class="slider-label">Height</p>
              <NSlider
                v-model:value="settings.data.settings.inpainting.height"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.inpainting.height"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="128"
                :max="2048"
              />
            </div>

            <!-- Steps -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Steps</p>
                </template>
                Number of steps to take in the diffusion process. Higher values
                will result in more detailed images but will take longer to
                generate. There is also a point of diminishing returns around
                100 steps.
                <b class="highlight"
                  >We recommend using 20-50 steps for most images.</b
                >
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.inpainting.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.inpainting.steps"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="5"
                :max="300"
              />
            </div>

            <CFGScale tab="inpainting" />

            <SAGInput tab="inpainting" />

            <!-- Number of images -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Batch Count</p>
                </template>
                Number of images to generate after each other.
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.inpainting.batch_count"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.inpainting.batch_count"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Batch Size</p>
                </template>
                Number of images to generate in paralel.
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.inpainting.batch_size"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.inpainting.batch_size"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>

            <!-- Seed -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Seed</p>
                </template>
                Seed is a number that represents the starting canvas of your
                image. If you want to create the same image as your friend, you
                can use the same settings and seed to do so.
                <b class="highlight">For random seed use -1.</b>
              </NTooltip>
              <NInputNumber
                v-model:value="settings.data.settings.inpainting.seed"
                size="small"
                :min="-1"
                :max="999_999_999_999"
                style="flex-grow: 1"
              />
            </div>
          </NSpace>
        </NCard>
      </NGi>

      <!-- Split -->

      <!-- Images -->
      <NGi>
        <GenerateSection :generate="generate" />

        <ImageOutput
          :current-image="global.state.inpainting.currentImage"
          :images="global.state.inpainting.images"
          :data="settings.data.settings.inpainting"
          @image-clicked="global.state.inpainting.currentImage = $event"
        />

        <OutputStats
          style="margin-top: 12px"
          :gen-data="global.state.inpainting.genData"
        />
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import { BurnerClock } from "@/clock";
import {
  GenerateSection,
  ImageOutput,
  OutputStats,
  Prompt,
  SamplerPicker,
  CFGScale,
  SAGInput,
} from "@/components";
import { serverUrl } from "@/env";
import {
  ArrowRedoSharp,
  ArrowUndoSharp,
  BrushSharp,
  TrashBinSharp,
} from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NGi,
  NGrid,
  NIcon,
  NInputNumber,
  NSlider,
  NSpace,
  NTooltip,
  useMessage,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { onUnmounted, ref } from "vue";
import VueDrawingCanvas from "vue-drawing-canvas";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const settings = useSettings();
const messageHandler = useMessage();

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999_999_999_999);
  }

  return seed;
};

const generate = () => {
  if (settings.data.settings.inpainting.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }

  generateMask();

  global.state.generating = true;

  const seed = checkSeed(settings.data.settings.inpainting.seed);

  fetch(`${serverUrl}/api/generate/inpainting`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        prompt: settings.data.settings.inpainting.prompt,
        image: settings.data.settings.inpainting.image,
        mask_image: settings.data.settings.inpainting.mask_image,
        id: uuidv4(),
        negative_prompt: settings.data.settings.inpainting.negative_prompt,
        width: settings.data.settings.inpainting.width,
        height: settings.data.settings.inpainting.height,
        steps: settings.data.settings.inpainting.steps,
        guidance_scale: settings.data.settings.inpainting.cfg_scale,
        seed: seed,
        batch_size: settings.data.settings.inpainting.batch_size,
        batch_count: settings.data.settings.inpainting.batch_count,
        scheduler: settings.data.settings.inpainting.sampler,
        self_attention_scale:
          settings.data.settings.inpainting.self_attention_scale,
        sigmas: settings.data.settings.inpainting.sigmas,
        sampler_settings:
          settings.data.settings.sampler_config[
            settings.data.settings.inpainting.sampler
          ],
        prompt_to_prompt_settings: {
          prompt_to_prompt_model:
            settings.data.settings.api.prompt_to_prompt_model,
          prompt_to_prompt_model_settings:
            settings.data.settings.api.prompt_to_prompt_device,
          prompt_to_prompt: settings.data.settings.api.prompt_to_prompt,
        },
      },
      model: settings.data.settings.model?.path,
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res.json().then((data) => {
        global.state.inpainting.images = data.images;
        global.state.inpainting.currentImage = data.images[0];
        global.state.progress = 0;
        global.state.total_steps = 0;
        global.state.current_step = 0;

        global.state.inpainting.genData = {
          time_taken: parseFloat(parseFloat(data.time as string).toFixed(4)),
          seed: seed,
        };
      });
    })
    .catch((err) => {
      global.state.generating = false;
      messageHandler.error(err);
    });
};

const canvas = ref<InstanceType<typeof VueDrawingCanvas>>();
const maskCanvas = ref<InstanceType<typeof VueDrawingCanvas>>();

const width = ref(512);
const height = ref(512);

const strokeWidth = ref(10);
const eraser = ref(false);
const preview = ref("");

const imageContainer = ref<HTMLElement>();

function handleImageUpdate(img: HTMLImageElement) {
  const containerWidth = imageContainer.value?.clientWidth;
  if (containerWidth === undefined) return;

  // Scale to fit container and keep aspect ratio
  const containerScaledWidth = containerWidth;
  const containerScaledHeight = (img.height * containerScaledWidth) / img.width;

  // Scale to fit into 70vh of the screen
  const screenHeight = window.innerHeight;
  const screenHeightScaledHeight =
    (containerScaledHeight * 0.7 * screenHeight) / containerScaledHeight;

  // Scale width to keep aspect ratio
  const screenHeightScaledWidth =
    (img.width * screenHeightScaledHeight) / img.height;

  // Select smaller of the two to fit into the container
  if (containerScaledWidth < screenHeightScaledWidth) {
    width.value = containerScaledWidth;
    height.value = containerScaledHeight;
  } else {
    width.value = screenHeightScaledWidth;
    height.value = screenHeightScaledHeight;
  }

  canvas.value?.redraw(false);
}

function previewImage(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const i = e.target?.result;
      if (i) {
        const s = i.toString();
        preview.value = s;
        const img = new Image();
        img.src = s;
        img.onload = () => {
          handleImageUpdate(img);
          settings.data.settings.inpainting.image = s;
        };
      }
    };
    reader.readAsDataURL(input.files[0]);
  }
}

async function clearCanvas() {
  canvas.value?.reset();
}

function undo() {
  canvas.value?.undo();
}

function redo() {
  canvas.value?.redo();
}

function toggleEraser() {
  eraser.value = !eraser.value;
}

function generateMask() {
  const x = canvas.value?.getAllStrokes();
  if (maskCanvas.value !== undefined) {
    maskCanvas.value.images = x;
    maskCanvas.value.redraw(true);
  }
}

// Burner clock
const burner = new BurnerClock(
  settings.data.settings.inpainting,
  settings,
  generate
);
onUnmounted(() => {
  burner.cleanup();
});

if (settings.data.settings.inpainting.image !== "") {
  preview.value = settings.data.settings.inpainting.image;
  const img = new Image();
  img.src = settings.data.settings.inpainting.image;
  img.onload = () => {
    handleImageUpdate(img);
  };
}
</script>
<style scoped>
.hidden-input {
  display: none;
}

.utility-button {
  margin-right: 8px;
}

.file-upload {
  appearance: none;
  background-color: transparent;
  border: 1px solid #63e2b7;
  border-radius: 6px;
  box-shadow: rgba(27, 31, 35, 0.1) 0 1px 0;
  box-sizing: border-box;
  color: #63e2b7;
  cursor: pointer;
  display: inline-block;
  padding: 6px 16px;
  position: relative;
  text-align: center;
  text-decoration: none;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: middle;
  white-space: nowrap;
}

.file-upload:focus:not(:focus-visible):not(.focus-visible) {
  box-shadow: none;
  outline: none;
}

.file-upload:focus {
  box-shadow: rgba(46, 164, 79, 0.4) 0 0 0 3px;
  outline: none;
}

.file-upload:disabled {
  background-color: #94d3a2;
  border-color: rgba(27, 31, 35, 0.1);
  color: rgba(255, 255, 255, 0.8);
  cursor: default;
}

.image-container {
  width: 100%;
  display: flex;
  justify-content: center;
}
</style>
