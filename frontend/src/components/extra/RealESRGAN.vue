<template>
  <div style="margin: 0 12px">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <NButton @click="loadUpscaler"> Load Upscaler </NButton>

        <ImageUpload
          :callback="imageSelectCallback"
          :preview="conf.data.settings.realesrgan.image"
          style="margin-bottom: 12px"
          @file-dropped="conf.data.settings.realesrgan.image = $event"
        />

        <NCard title="Settings">
          <NSpace vertical class="left-container">
            <!-- Scale factor -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Scale Factor</p>
                </template>
                TODO
              </NTooltip>
              <NSlider
                v-model:value="conf.data.settings.realesrgan.scale_factor"
                :min="2"
                :max="4"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="conf.data.settings.realesrgan.scale_factor"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="2"
                :max="4"
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
          :current-image="global.state.extra.currentImage"
          :images="global.state.extra.images"
          @image-clicked="global.state.extra.currentImage = $event"
        />
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import GenerateSection from "@/components/GenerateSection.vue";
import ImageOutput from "@/components/ImageOutput.vue";
import ImageUpload from "@/components/ImageUpload.vue";
import { serverUrl } from "@/env";
import {
  NButton,
  NCard,
  NGi,
  NGrid,
  NInputNumber,
  NSlider,
  NSpace,
  NTooltip,
  useMessage,
} from "naive-ui";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const conf = useSettings();
const messageHandler = useMessage();

const imageSelectCallback = (base64Image: string) => {
  conf.data.settings.realesrgan.image = base64Image;
};

const loadUpscaler = () => {
  const url = new URL(`${serverUrl}/api/models/load`);
  url.searchParams.append("model", conf.data.settings.realesrgan.model);
  url.searchParams.append("backend", "PyTorch");

  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => {
      res.json().then((data) => {
        if (data.error) {
          messageHandler.error(data.error);
        } else {
          messageHandler.success("Upscaler loaded");
        }
      });
    })
    .catch((err) => {
      messageHandler.error(err);
      console.log(err);
    });
};

const generate = () => {
  global.state.generating = true;
  fetch(`${serverUrl}/api/generate/realesrgan-upscale`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        image: conf.data.settings.realesrgan.image,
        scale_factor: conf.data.settings.realesrgan.scale_factor,
        model: conf.data.settings.realesrgan.model,
      },
      model: conf.data.settings.realesrgan.model,
    }),
  })
    .then((res) => {
      global.state.generating = false;
      res.json().then((data) => {
        console.log(data);
        global.state.extra.images = data.images;
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
