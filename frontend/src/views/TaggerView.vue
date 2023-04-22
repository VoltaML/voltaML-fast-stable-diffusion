<template>
  <div class="main-container">
    <!-- Main -->
    <NGrid cols="1 850:2" x-gap="12">
      <NGi>
        <ImageUpload
          :callback="imageSelectCallback"
          :preview="conf.data.settings.tagger.image"
          style="margin-bottom: 12px"
          @file-dropped="conf.data.settings.tagger.image = $event"
        />

        <NCard title="Settings">
          <NSpace vertical class="left-container">
            <!-- Sampler -->
            <div class="flex-container">
              <p class="slider-label">Sampler</p>

              <NSelect
                :options="[
                  {
                    label: 'Deepdanbooru',
                    value: 'deepdanbooru',
                  },
                  {
                    label: 'CLIP',
                    value: 'clip',
                  },
                  {
                    label: 'Flamingo',
                    value: 'flamingo',
                  },
                ]"
                v-model:value="conf.data.settings.tagger.model"
                style="flex-grow: 1"
              />
            </div>

            <!-- Treshold -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Treshold</p>
                </template>
                Confidence treshold of the model. The higher the value, the more
                tokens will be given to you.
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.tagger.treshold"
                :min="0.1"
                :max="1"
                style="margin-right: 12px"
                :step="0.025"
              />
              <NInputNumber
                v-model:value="conf.data.settings.tagger.treshold"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="0.1"
                :max="1"
                :step="0.025"
              />
            </div>
          </NSpace>
        </NCard>
      </NGi>

      <!-- Split -->

      <!-- Images -->
      <NGi>
        <GenerateSection :generate="generate" />
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import GenerateSection from "@/components/GenerateSection.vue";
import ImageUpload from "@/components/ImageUpload.vue";
import { serverUrl } from "@/env";
import {
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

const global = useState();
const conf = useSettings();
const messageHandler = useMessage();

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999_999_999_999);
  }

  return seed;
};

const imageSelectCallback = (base64Image: string) => {
  conf.data.settings.img2img.image = base64Image;
};

const generate = () => {
  if (conf.data.settings.img2img.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }
  global.state.generating = true;
  fetch(`${serverUrl}/api/generate/img2img`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        prompt: conf.data.settings.img2img.prompt,
        image: conf.data.settings.img2img.image,
        id: uuidv4(),
        negative_prompt: conf.data.settings.img2img.negative_prompt,
        width: conf.data.settings.img2img.width,
        height: conf.data.settings.img2img.height,
        steps: conf.data.settings.img2img.steps,
        guidance_scale: conf.data.settings.img2img.cfg_scale,
        seed: checkSeed(conf.data.settings.img2img.seed),
        batch_size: conf.data.settings.img2img.batch_size,
        batch_count: conf.data.settings.img2img.batch_count,
        strength: conf.data.settings.img2img.denoising_strength,
        scheduler: conf.data.settings.img2img.sampler,
      },
      model: conf.data.settings.model?.name,
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res.json().then((data) => {
        global.state.img2img.images = data.images;
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
