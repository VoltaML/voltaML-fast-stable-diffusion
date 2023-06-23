<template>
  <div style="margin: 0 12px">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <ImageUpload
          :callback="imageSelectCallback"
          :preview="conf.data.settings.sd_upscale.image"
          style="margin-bottom: 12px"
          @file-dropped="conf.data.settings.sd_upscale.image = $event"
        />

        <NCard title="Settings" style="margin-bottom: 12px">
          <NSpace vertical class="left-container">
            <!-- Prompt -->
            <NInput
              v-model:value="conf.data.settings.sd_upscale.prompt"
              type="textarea"
              placeholder="Prompt"
              show-count
              @keyup="
                promptHandleKeyUp(
                  $event,
                  conf.data.settings.sd_upscale,
                  'prompt'
                )
              "
              @keydown="promptHandleKeyDown"
            >
              <template #count>{{ promptCount }}</template>
            </NInput>
            <NInput
              v-model:value="conf.data.settings.sd_upscale.negative_prompt"
              type="textarea"
              placeholder="Negative prompt"
              show-count
              @keyup="
                promptHandleKeyUp(
                  $event,
                  conf.data.settings.sd_upscale,
                  'negative_prompt'
                )
              "
              @keydown="promptHandleKeyDown"
            >
              <template #count>{{ negativePromptCount }}</template>
            </NInput>

            <!-- Sampler -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p style="margin-right: 12px; width: 150px">Sampler</p>
                </template>
                The sampler is the method used to generate the image. Your
                result may vary drastically depending on the sampler you choose.
                <b class="highlight"
                  >We recommend using DPMSolverMultistep for the best results .
                </b>
                <a
                  target="_blank"
                  href="https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
                  >Learn more</a
                >
              </NTooltip>

              <NSelect
                :options="conf.scheduler_options"
                v-model:value="conf.data.settings.sd_upscale.sampler"
                style="flex-grow: 1"
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
                v-model:value="conf.data.settings.sd_upscale.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.steps"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="5"
                :max="300"
              />
            </div>

            <!-- CFG Scale -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">CFG Scale</p>
                </template>
                Guidance scale indicates how much should model stay close to the
                prompt. Higher values might be exactly what you want, but
                generated images might have some artefacts. Lower values
                indicates that model can "dream" about this prompt more.
                <b class="highlight"
                  >We recommend using 3-15 for most images.</b
                >
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.sd_upscale.cfg_scale"
                :min="1"
                :max="30"
                :step="0.5"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.cfg_scale"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="30"
                :step="0.5"
              />
            </div>

            <!-- Number of images -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Batch Count</p>
                </template>
                Number of images to generate after each other.
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.sd_upscale.batch_count"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.batch_count"
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
                v-model:value="conf.data.settings.sd_upscale.batch_size"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.batch_size"
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
                v-model:value="conf.data.settings.sd_upscale.seed"
                size="small"
                :min="-1"
                :max="999_999_999_999"
                style="flex-grow: 1"
              />
            </div>

            <!-- Tile Size -->
            <div class="flex-container">
              <p class="slider-label">Tile Size</p>
              <NSlider
                v-model:value="conf.data.settings.sd_upscale.tile_size"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.tile_size"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="128"
                :max="2048"
              />
            </div>

            <!-- Tile Border -->
            <div class="flex-container">
              <p class="slider-label">Tile Border</p>
              <NSlider
                v-model:value="conf.data.settings.sd_upscale.tile_border"
                :min="8"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.tile_border"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="8"
                :max="2048"
              />
            </div>

            <!-- Original Image Slice -->
            <div class="flex-container">
              <p class="slider-label">Original Image Slice</p>
              <NSlider
                v-model:value="
                  conf.data.settings.sd_upscale.original_image_slice
                "
                :min="8"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="
                  conf.data.settings.sd_upscale.original_image_slice
                "
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="8"
                :max="2048"
              />
            </div>

            <!-- Noise level -->
            <div class="flex-container">
              <p class="slider-label">Noise Level</p>
              <NSlider
                v-model:value="conf.data.settings.sd_upscale.noise_level"
                :min="1"
                :max="100"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.sd_upscale.noise_level"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="100"
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
          :current-image="global.state.sd_upscale.currentImage"
          :images="global.state.sd_upscale.images"
          @image-clicked="global.state.sd_upscale.currentImage = $event"
        />

        <OutputStats
          style="margin-top: 12px"
          :gen-data="global.state.sd_upscale.genData"
        />
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import { BurnerClock } from "@/clock";
import GenerateSection from "@/components/GenerateSection.vue";
import ImageOutput from "@/components/ImageOutput.vue";
import ImageUpload from "@/components/ImageUpload.vue";
import OutputStats from "@/components/OutputStats.vue";
import { serverUrl } from "@/env";
import {
  promptHandleKeyDown,
  promptHandleKeyUp,
  spaceRegex,
} from "@/functions";
import {
  NCard,
  NGi,
  NGrid,
  NInput,
  NInputNumber,
  NSelect,
  NSlider,
  NSpace,
  NTooltip,
  useMessage,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { computed, onUnmounted } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const conf = useSettings();
const messageHandler = useMessage();

const promptCount = computed(() => {
  return conf.data.settings.sd_upscale.prompt.split(spaceRegex).length - 1;
});
const negativePromptCount = computed(() => {
  return (
    conf.data.settings.sd_upscale.negative_prompt.split(spaceRegex).length - 1
  );
});

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999_999_999_999);
  }

  return seed;
};

const imageSelectCallback = (base64Image: string) => {
  conf.data.settings.sd_upscale.image = base64Image;
};

const generate = () => {
  if (conf.data.settings.sd_upscale.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }

  global.state.generating = true;

  const seed = checkSeed(conf.data.settings.sd_upscale.seed);

  fetch(`${serverUrl}/api/generate/sd-upscale`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        prompt: conf.data.settings.sd_upscale.prompt,
        image: conf.data.settings.sd_upscale.image,
        scheduler: conf.data.settings.sd_upscale.sampler,
        id: uuidv4(),
        negative_prompt: conf.data.settings.sd_upscale.negative_prompt,
        steps: conf.data.settings.sd_upscale.steps,
        guidance_scale: conf.data.settings.sd_upscale.cfg_scale,
        seed: seed,
        batch_size: conf.data.settings.sd_upscale.batch_size,
        batch_count: conf.data.settings.sd_upscale.batch_count,
        tile_size: conf.data.settings.sd_upscale.tile_size,
        tile_border: conf.data.settings.sd_upscale.tile_border,
        original_image_slice:
          conf.data.settings.sd_upscale.original_image_slice,
        noise_level: conf.data.settings.sd_upscale.noise_level,
      },
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res.json().then((data) => {
        global.state.sd_upscale.images = data.images;
        global.state.sd_upscale.currentImage = data.images[0];
        global.state.progress = 0;
        global.state.total_steps = 0;
        global.state.current_step = 0;

        global.state.sd_upscale.genData = {
          time_taken: parseFloat(parseFloat(data.time as string).toFixed(4)),
          seed: seed,
        };
      });
    })
    .catch((err) => {
      global.state.generating = false;
      messageHandler.error(err);
      console.log(err);
    });
};

// Burner clock
const burner = new BurnerClock(conf.data.settings.sd_upscale, conf, generate);
onUnmounted(() => {
  burner.cleanup();
});
</script>
<style scoped>
.image-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  overflow: hidden;
}

.image-container {
  height: 70vh;
  width: 100%;
  display: flex;
  justify-content: center;
}
</style>
