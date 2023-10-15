<template>
  <div style="margin: 0 12px">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <ImageUpload
          :callback="imageSelectCallback"
          :preview="settings.data.settings.upscale.image"
          style="margin-bottom: 12px"
          @file-dropped="settings.data.settings.upscale.image = $event"
        />

        <NCard title="Settings" style="margin-bottom: 12px">
          <NSpace vertical class="left-container">
            <!-- Upscaler model -->
            <div class="flex-container">
              <p class="slider-label">Model</p>
              <NSelect
                v-model:value="settings.data.settings.upscale.model"
                style="margin-right: 12px"
                filterable
                tag
                :options="upscalerOptionsFull"
              />
            </div>

            <!-- Scale factor -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Scale Factor</p>
                </template>
                TODO
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.upscale.upscale_factor"
                :min="1"
                :max="4"
                :step="0.1"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.upscale.upscale_factor"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="4"
                :step="0.1"
              />
            </div>

            <!-- Tile Size -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Tile Size</p>
                </template>
                How large each tile should be. Larger tiles will use more
                memory. 0 will disable tiling.
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.upscale.tile_size"
                :min="32"
                :max="2048"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.upscale.tile_size"
                size="small"
                :min="32"
                :max="2048"
                style="min-width: 96px; width: 96px"
              />
            </div>

            <!-- Tile Padding -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Tile Padding</p>
                </template>
                How much should tiles overlap. Larger padding will use more
                memory, but image should not have visible seams.
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.upscale.tile_padding"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.upscale.tile_padding"
                size="small"
                style="min-width: 96px; width: 96px"
              />
            </div>
          </NSpace>
        </NCard>
      </NGi>

      <!-- Split -->

      <!-- Images -->
      <NGi>
        <GenerateSection :generate="generate" do-not-disable-generate />

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
import type { SelectMixedOption } from "naive-ui/es/select/src/interface";
import { computed } from "vue";
import { upscalerOptions, useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const settings = useSettings();
const messageHandler = useMessage();

const imageSelectCallback = (base64Image: string) => {
  settings.data.settings.upscale.image = base64Image;
};

const upscalerOptionsFull = computed<SelectMixedOption[]>(() => {
  const localModels = global.state.models
    .filter(
      (model) =>
        model.backend === "Upscaler" &&
        !(
          upscalerOptions
            .map((option: SelectMixedOption) => option.label)
            .indexOf(model.name) !== -1
        )
    )
    .map((model) => ({
      label: model.name,
      value: model.path,
    }));

  return [...upscalerOptions, ...localModels];
});

const generate = () => {
  global.state.generating = true;
  fetch(`${serverUrl}/api/generate/upscale`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        image: settings.data.settings.upscale.image,
        upscale_factor: settings.data.settings.upscale.upscale_factor,
        model: settings.data.settings.upscale.model,
        tile_size: settings.data.settings.upscale.tile_size,
        tile_padding: settings.data.settings.upscale.tile_padding,
      },
      model: settings.data.settings.upscale.model,
    }),
  })
    .then((res) => {
      global.state.generating = false;
      res.json().then((data) => {
        console.log(data);
        global.state.extra.images = [data.images];
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
