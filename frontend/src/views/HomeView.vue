<template>
  <div class="main-container">
    <!-- Main -->

    <NGrid cols="2" x-gap="12">
      <NGi>
        <NSpace vertical class="left-container">
          <!-- Backend selection -->
          <NSpace>
            <NTooltip :max-width="600">
              <template #trigger>
                <p>Backend</p>
              </template>
              <b class="highlight">TensorRT is the fastest</b> method but
              requires accelerated model that must be created per machine.
              <b class="highlight">PyTorch</b> on the other hand is
              <b class="highlight">slower but works right out of the box.</b>
            </NTooltip>
            <NSpace align="center" style="height: 100%">
              <NRadioGroup
                v-model:value="conf.data.settings.backend"
                name="backend-selection"
              >
                <NRadioButton
                  v-for="backend in [
                    { label: 'TensorRT', value: 'TensorRT' },
                    { label: 'PyTorch', value: 'PyTorch' },
                  ]"
                  :key="backend.value"
                  :value="backend.value"
                  :label="backend.label"
                />
              </NRadioGroup>
            </NSpace>
          </NSpace>

          <!-- Prompt -->
          <NInput
            v-model:value="conf.data.settings.txt2img.prompt"
            type="textarea"
            placeholder="Prompt"
          />
          <NInput
            v-model:value="conf.data.settings.txt2img.negativePrompt"
            type="textarea"
            placeholder="Negative prompt"
          />

          <!-- Sampler -->
          <div class="flex-container">
            <NTooltip :max-width="600">
              <template #trigger>
                <p style="margin-right: 12px">Sampler</p>
              </template>
              The sampler is the method used to generate the image. Your result
              may vary drastically depending on the sampler you choose.
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
              :options="[
                { label: 'Euler A', value: 'Euler A' },
                { label: 'Euler', value: 'Euler' },
                { label: 'DDIM', value: 'DDIM' },
                { label: 'Heun', value: 'Heun' },
                { label: 'DPM Dicsrete', value: 'DPM Dicsrete' },
                { label: 'DPM A', value: 'DPM A' },
                { label: 'LMS', value: 'LMS' },
                { label: 'PNDM', value: 'PNDM' },
                { label: 'DPMPP SDE A', value: 'DPMPP SDE A' },
                { label: 'DPMPP 2M', value: 'DPMPP 2M' },
              ]"
              v-model:value="conf.data.settings.txt2img.sampler"
              style="flex-grow: 1"
            />
          </div>

          <!-- Dimensions -->
          <div class="flex-container">
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
            />
          </div>
          <div class="flex-container">
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
              generate. There is also a point of diminishing returns around 100
              steps.
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
            <NTooltip :max-width="600">
              <template #trigger>
                <p class="slider-label">CFG Scale</p>
              </template>
              Guidance scale indicates how much should model stay close to the
              prompt. Higher values might be exactly what you want, but
              generated images might have some artefacts. Lower values indicates
              that model can "dream" about this prompt more.
              <b class="highlight">We recommend using 3-15 for most images.</b>
            </NTooltip>
            <NSlider
              v-model:value="conf.data.settings.txt2img.cfgScale"
              :min="1"
              :max="30"
              :step="0.5"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="conf.data.settings.txt2img.cfgScale"
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
              v-model:value="conf.data.settings.txt2img.batchCount"
              :min="1"
              :max="9"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="conf.data.settings.txt2img.batchCount"
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
              v-model:value="conf.data.settings.txt2img.seed"
              size="small"
              :min="-1"
              :max="999999999"
              style="min-width: 96px; width: 96px"
            />
          </div>

          <!-- Generate button -->
          <NSpace justify="center">
            <NButton
              type="success"
              @click="generate"
              :disabled="global.state.generating"
              >Generate</NButton
            >
          </NSpace>
        </NSpace>
      </NGi>

      <!-- Split -->

      <!-- Images -->
      <NGi>
        <NSpace justify="center">
          <NImage
            v-if="global.state.txt2img.currentImage"
            :src="`data:image/png;base64,${global.state.txt2img.currentImage}`"
            :img-props="{
              style: 'max-width: 100%; max-height: 70vh; object-fit: contain;',
            }"
          />
        </NSpace>
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import { serverUrl } from "@/env";
import {
  NButton,
  NGi,
  NGrid,
  NImage,
  NInput,
  NInputNumber,
  NRadioButton,
  NRadioGroup,
  NSelect,
  NSlider,
  NSpace,
  NTooltip,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";

const global = useState();
const conf = useSettings();

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999999999);
  }

  return seed;
};

const generate = () => {
  global.state.generating = true;
  fetch(`${serverUrl}/api/txt2img/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        id: uuidv4(),
        prompt: conf.data.settings.txt2img.prompt,
        negative_prompt: conf.data.settings.txt2img.negativePrompt,
        width: conf.data.settings.txt2img.width,
        height: conf.data.settings.txt2img.height,
        steps: conf.data.settings.txt2img.steps,
        guidance_scale: conf.data.settings.txt2img.cfgScale,
        seed: checkSeed(conf.data.settings.txt2img.seed),
        batch_size: 1,
        batch_count: conf.data.settings.txt2img.batchCount,
      },
      model: "Linaqruf/anything-v3.0",
      scheduler: "Euler A",
      backend: "PyTorch",
      autoload: false,
    }),
  })
    .then((res) => {
      global.state.generating = false;
      res.json().then((data) => {
        global.state.txt2img.currentImage = data.images[0];
        global.state.progress = 0;
      });
    })
    .catch((err) => {
      global.state.generating = false;
      console.log(err);
    });
};
</script>

<style scoped>
.left-container {
  margin: 0 12px;
}

.split {
  width: 50%;
}

.flex-container {
  width: 100%;
  display: inline-flex;
  align-items: center;
}

.slider-label {
  margin-right: 12px;
  width: 64px;
}

.main-container {
  margin: 18px;
}

.highlight {
  color: #63e2b7;
}
</style>
