<template>
  <NAlert
    style="width: 100%; margin-bottom: 12px"
    type="warning"
    title="Does not work yet"
  />
  <div style="margin: 0 12px">
    <!-- Main -->
    <NGrid cols="1 850:2" x-gap="12">
      <NGi>
        <ImageUpload
          :callback="imageSelectCallback"
          :preview="conf.data.settings.imageVariations.image"
          style="margin-bottom: 12px"
        />

        <NCard title="Settings">
          <NSpace vertical class="left-container">
            <!-- Sampler -->
            <div class="flex-container">
              <NTooltip :max-width="600">
                <template #trigger>
                  <p style="margin-right: 12px; width: 150px">Sampler</p>
                </template>
                The sampler is the method used to generate the image. Your
                result may vary drastically depending on the sampler you choose.
                <b class="highlight"
                  >We recommend using Euler A for the best results (but it also
                  takes more time).
                </b>
                <a
                  target="_blank"
                  href="https://docs.google.com/document/d/1n0YozLAUwLJWZmbsx350UD_bwAx3gZMnRuleIZt_R1w"
                  >Learn more</a
                >
              </NTooltip>

              <NSelect
                :options="conf.scheduler_options"
                v-model:value="conf.data.settings.imageVariations.sampler"
                style="flex-grow: 1"
              />
            </div>

            <!-- Steps -->
            <div class="flex-container">
              <NTooltip :max-width="600">
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
                v-model:value="conf.data.settings.imageVariations.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.imageVariations.steps"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="5"
                :max="300"
              />
            </div>

            <!-- CFG Scale -->
            <div class="flex-container">
              <NTooltip :max-width="600">
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
                v-model:value="conf.data.settings.imageVariations.cfgScale"
                :min="1"
                :max="30"
                :step="0.5"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.imageVariations.cfgScale"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="30"
                :step="0.5"
              />
            </div>

            <!-- Number of images -->
            <div class="flex-container">
              <NTooltip :max-width="600">
                <template #trigger>
                  <p class="slider-label">Batch Count</p>
                </template>
                Number of images to generate after each other.
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.imageVariations.batchCount"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.imageVariations.batchCount"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>
            <div class="flex-container">
              <NTooltip :max-width="600">
                <template #trigger>
                  <p class="slider-label">Batch Size</p>
                </template>
                Number of images to generate in paralel.
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.imageVariations.batchSize"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.imageVariations.batchSize"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>

            <!-- Seed -->
            <div class="flex-container">
              <NTooltip :max-width="600">
                <template #trigger>
                  <p class="slider-label">Seed</p>
                </template>
                Seed is a number that represents the starting canvas of your
                image. If you want to create the same image as your friend, you
                can use the same settings and seed to do so.
                <b class="highlight">For random seed use -1.</b>
              </NTooltip>
              <NInputNumber
                v-model:value="conf.data.settings.imageVariations.seed"
                size="small"
                :min="-1"
                :max="999999999"
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
          :current-image="global.state.imageVariations.currentImage"
          :images="global.state.imageVariations.images"
        />
      </NGi>
    </NGrid>
  </div>
</template>

<script lang="ts" setup>
import ImageOutput from "@/components/ImageOutput.vue";
import { serverUrl } from "@/env";
import {
  NAlert,
  NCard,
  NGi,
  NGrid,
  NInputNumber,
  NSelect,
  NSlider,
  NSpace,
  NTooltip,
  useMessage,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";
import GenerateSection from "./GenerateSection.vue";
import ImageUpload from "./ImageUpload.vue";

const global = useState();
const conf = useSettings();
const messageHandler = useMessage();

const imageSelectCallback = (base64Image: string) => {
  conf.data.settings.imageVariations.image = base64Image;
};

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999999999);
  }

  return seed;
};

const generate = () => {
  if (conf.data.settings.imageVariations.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }
  global.state.generating = true;
  fetch(`${serverUrl}/api/generate/image_variations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        image: conf.data.settings.imageVariations.image,
        id: uuidv4(),
        steps: conf.data.settings.imageVariations.steps,
        guidance_scale: conf.data.settings.imageVariations.cfgScale,
        seed: checkSeed(conf.data.settings.imageVariations.seed),
        batch_size: conf.data.settings.imageVariations.batchSize,
        batch_count: conf.data.settings.imageVariations.batchCount,
        scheduler: conf.data.settings.imageVariations.sampler,
      },
      model: conf.data.settings.model,
    }),
  })
    .then((res) => {
      global.state.generating = false;
      res.json().then((data) => {
        global.state.imageVariations.images = data.images;
        global.state.progress = 0;
        global.state.total_steps = 0;
        global.state.current_step = 0;
      });
    })
    .catch((err) => {
      global.state.generating = false;
      messageHandler.error(err);
      console.log(err);
    });
};
</script>
