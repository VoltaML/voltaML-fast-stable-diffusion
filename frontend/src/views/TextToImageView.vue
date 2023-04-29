<template>
  <div class="main-container">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <NCard title="Settings">
          <NSpace vertical class="left-container">
            <!-- Prompt -->
            <NInput
              v-model:value="conf.data.settings.txt2img.prompt"
              type="textarea"
              placeholder="Prompt"
              show-count
            >
              <template #count>{{ promptCount }}</template>
            </NInput>
            <NInput
              v-model:value="conf.data.settings.txt2img.negative_prompt"
              type="textarea"
              placeholder="Negative prompt"
              show-count
            >
              <template #count>{{ negativePromptCount }}</template>
            </NInput>

            <!-- Sampler -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p style="margin-right: 12px; width: 100px">Sampler</p>
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
                v-model:value="conf.data.settings.txt2img.sampler"
                style="flex-grow: 1"
              />
            </div>

            <!-- Dimensions -->
            <div class="flex-container" v-if="conf.data.settings.aitDim.width">
              <p class="slider-label">Width</p>
              <NSlider
                :value="conf.data.settings.aitDim.width"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                :value="conf.data.settings.aitDim.width"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="128"
                :max="2048"
              />
            </div>
            <div class="flex-container" v-else>
              <p class="slider-label">Width</p>
              <NSlider
                v-model:value="conf.data.settings.txt2img.width"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.width"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="128"
                :max="2048"
              />
            </div>
            <div class="flex-container" v-if="conf.data.settings.aitDim.height">
              <p class="slider-label">Height</p>
              <NSlider
                :value="conf.data.settings.aitDim.height"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                :value="conf.data.settings.aitDim.height"
                size="small"
                style="min-width: 96px; width: 96px"
                :step="8"
                :min="128"
                :max="2048"
              />
            </div>
            <div class="flex-container" v-else>
              <p class="slider-label">Height</p>
              <NSlider
                v-model:value="conf.data.settings.txt2img.height"
                :min="128"
                :max="2048"
                :step="8"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.height"
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
                v-model:value="conf.data.settings.txt2img.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.steps"
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
                v-model:value="conf.data.settings.txt2img.cfg_scale"
                :min="1"
                :max="30"
                :step="0.5"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.cfg_scale"
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
                v-model:value="conf.data.settings.txt2img.batch_count"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.batch_count"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>
            <div
              class="flex-container"
              v-if="conf.data.settings.aitDim.batch_size"
            >
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Batch Size</p>
                </template>
                Number of images to generate in paralel.
              </NTooltip>
              <NSlider
                :value="conf.data.settings.aitDim.batch_size"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                :value="conf.data.settings.aitDim.batch_size"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>
            <div class="flex-container" v-else>
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Batch Size</p>
                </template>
                Number of images to generate in paralel.
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.txt2img.batch_size"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.batch_size"
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
                  <p style="margin-right: 12px; width: 75px">Seed</p>
                </template>
                Seed is a number that represents the starting canvas of your
                image. If you want to create the same image as your friend, you
                can use the same settings and seed to do so.
                <b class="highlight">For random seed use -1.</b>
              </NTooltip>
              <NInputNumber
                v-model:value="conf.data.settings.txt2img.seed"
                size="small"
                :min="-1"
                :max="999_999_999_999"
                style="flex-grow: 1"
              />
            </div>
          </NSpace>
        </NCard>

        <NCard title="Highres fix" style="margin-top: 12px">
          <div class="flex-container">
            <div class="slider-label">
              <p>Enabled</p>
            </div>
            <NSwitch v-model:value="global.state.txt2img.highres" />
          </div>

          <NSpace
            vertical
            class="left-container"
            v-if="global.state.txt2img.highres"
          >
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
                v-model:value="conf.data.settings.extra.highres.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.extra.highres.steps"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="5"
                :max="300"
              />
            </div>

            <!-- Scale -->
            <div class="flex-container">
              <p class="slider-label">Scale</p>
              <NSlider
                v-model:value="conf.data.settings.extra.highres.scale"
                :min="1"
                :max="8"
                :step="1"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.extra.highres.scale"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="8"
                :step="1"
              />
            </div>

            <!-- Denoising strength -->
            <div class="flex-container">
              <p class="slider-label">Strength</p>
              <NSlider
                v-model:value="conf.data.settings.extra.highres.strength"
                :min="0.1"
                :max="0.9"
                :step="0.05"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.extra.highres.strength"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="0.1"
                :max="0.9"
                :step="0.05"
              />
            </div>

            <div class="flex-container">
              <p class="slider-label">Antialiased</p>
              <NSwitch
                v-model:value="conf.data.settings.extra.highres.antialiased"
              />
            </div>

            <div class="flex-container">
              <p class="slider-label">Latent Mode</p>
              <NSelect
                v-model:value="
                  conf.data.settings.extra.highres.latent_scale_mode
                "
                size="small"
                style="flex-grow: 1"
                :options="[
                  { label: 'Nearest', value: 'nearest' },
                  { label: 'Nearest exact', value: 'nearest-exact' },
                  { label: 'Linear', value: 'linear' },
                  { label: 'Bilinear', value: 'bilinear' },
                  { label: 'Bicubic', value: 'bicubic' },
                ]"
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
          :current-image="global.state.txt2img.currentImage"
          :images="global.state.txt2img.images"
          @image-clicked="global.state.txt2img.currentImage = $event"
        />

        <SendOutputTo :output="global.state.txt2img.currentImage" />

        <OutputStats
          style="margin-top: 12px"
          :gen-data="global.state.txt2img.genData"
        />
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import GenerateSection from "@/components/GenerateSection.vue";
import ImageOutput from "@/components/ImageOutput.vue";
import OutputStats from "@/components/OutputStats.vue";
import SendOutputTo from "@/components/SendOutputTo.vue";
import { serverUrl } from "@/env";
import { spaceRegex } from "@/functions";
import {
  NCard,
  NGi,
  NGrid,
  NInput,
  NInputNumber,
  NSelect,
  NSlider,
  NSpace,
  NSwitch,
  NTooltip,
  useMessage,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { computed } from "vue";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";

const global = useState();
const conf = useSettings();
const messageHandler = useMessage();

const promptCount = computed(() => {
  return conf.data.settings.txt2img.prompt.split(spaceRegex).length - 1;
});
const negativePromptCount = computed(() => {
  return (
    conf.data.settings.txt2img.negative_prompt.split(spaceRegex).length - 1
  );
});

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999_999_999_999);
  }

  return seed;
};

const generate = () => {
  if (conf.data.settings.txt2img.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }
  global.state.generating = true;

  const seed = checkSeed(conf.data.settings.txt2img.seed);

  fetch(`${serverUrl}/api/generate/txt2img`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        id: uuidv4(),
        prompt: conf.data.settings.txt2img.prompt,
        negative_prompt: conf.data.settings.txt2img.negative_prompt,
        width: conf.data.settings.aitDim.width
          ? conf.data.settings.aitDim.width
          : conf.data.settings.txt2img.width,
        height: conf.data.settings.aitDim.height
          ? conf.data.settings.aitDim.height
          : conf.data.settings.txt2img.height,
        steps: conf.data.settings.txt2img.steps,
        guidance_scale: conf.data.settings.txt2img.cfg_scale,
        seed: seed,
        batch_size: conf.data.settings.aitDim.batch_size
          ? conf.data.settings.aitDim.batch_size
          : conf.data.settings.txt2img.batch_size,
        batch_count: conf.data.settings.txt2img.batch_count,
        scheduler: conf.data.settings.txt2img.sampler,
      },
      model: conf.data.settings.model?.name,
      backend: "PyTorch",
      autoload: false,
      flags: global.state.txt2img.highres
        ? {
            highres_fix: {
              scale: conf.data.settings.extra.highres.scale,
              latent_scale_mode:
                conf.data.settings.extra.highres.latent_scale_mode,
              strength: conf.data.settings.extra.highres.strength,
              steps: conf.data.settings.extra.highres.steps,
              antialiased: conf.data.settings.extra.highres.antialiased,
            },
          }
        : {},
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res.json().then((data) => {
        global.state.txt2img.images = data.images;
        global.state.progress = 0;
        global.state.total_steps = 0;
        global.state.current_step = 0;

        global.state.txt2img.genData = {
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
</script>
