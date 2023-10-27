<template>
  <div style="margin: 0 12px">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <ImageUpload
          :callback="imageSelectCallback"
          :preview="settings.data.settings.controlnet.image"
          style="margin-bottom: 12px"
          @file-dropped="settings.data.settings.controlnet.image = $event"
        />

        <NCard title="Settings" style="margin-bottom: 12px">
          <NSpace vertical class="left-container">
            <!-- Prompt -->
            <NInput
              v-model:value="settings.data.settings.controlnet.prompt"
              type="textarea"
              placeholder="Prompt"
              show-count
              @keyup="
                promptHandleKeyUp(
                  $event,
                  settings.data.settings.controlnet,
                  'prompt',
                  global
                )
              "
              @keydown="promptHandleKeyDown"
            >
              <template #count>{{ promptCount }}</template>
            </NInput>
            <NInput
              v-model:value="settings.data.settings.controlnet.negative_prompt"
              type="textarea"
              placeholder="Negative prompt"
              show-count
              @keyup="
                promptHandleKeyUp(
                  $event,
                  settings.data.settings.controlnet,
                  'negative_prompt',
                  global
                )
              "
              @keydown="promptHandleKeyDown"
            >
              <template #count>{{ negativePromptCount }}</template>
            </NInput>

            <!-- Sampler -->
            <SamplerPicker type="controlnet" />

            <!-- ControlNet mode -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p style="margin-right: 12px; width: 150px">ControlNet</p>
                </template>
                ControlNet is a method of guiding the diffusion process. It
                allows you to control the output by providing a guidance image.
                This image will be processed automatically. You can also opt out
                and enable "Is Preprocessed" to provide your own preprocessed
                image.
                <a
                  href="https://github.com/lllyasviel/ControlNet-v1-1-nightly?tab=readme-ov-file#controlnet-11"
                  >Learn more</a
                >
              </NTooltip>

              <NSelect
                :options="settings.controlnet_options"
                v-model:value="settings.data.settings.controlnet.controlnet"
                filterable
                tag
                style="flex-grow: 1"
              />
            </div>

            <DimensionsInput
              :dimensions-object="settings.data.settings.controlnet"
            />

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
                v-model:value="settings.data.settings.controlnet.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.controlnet.steps"
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
                v-model:value="settings.data.settings.controlnet.cfg_scale"
                :min="1"
                :max="30"
                :step="0.5"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.controlnet.cfg_scale"
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
                v-model:value="settings.data.settings.controlnet.batch_count"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.controlnet.batch_count"
                size="small"
                style="min-width: 96px; width: 96px"
                :min="1"
                :max="9"
              />
            </div>

            <BatchSizeInput
              :batch-size-object="settings.data.settings.controlnet"
            />

            <!-- ControlNet Conditioning Scale -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">ControlNet Conditioning Scale</p>
                </template>
                How much should the ControlNet affect the image.
              </NTooltip>
              <NSlider
                v-model:value="
                  settings.data.settings.controlnet
                    .controlnet_conditioning_scale
                "
                :min="0.1"
                :max="2"
                style="margin-right: 12px"
                :step="0.025"
              />
              <NInputNumber
                v-model:value="
                  settings.data.settings.controlnet
                    .controlnet_conditioning_scale
                "
                size="small"
                style="min-width: 96px; width: 96px"
                :min="0.1"
                :max="2"
                :step="0.025"
              />
            </div>

            <!-- Detection resolution -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Detection resolution</p>
                </template>
                What resolution to use for the image processing. This process
                does not affect the final result but can affect the quality of
                the ControlNet processing.
              </NTooltip>
              <NSlider
                v-model:value="
                  settings.data.settings.controlnet.detection_resolution
                "
                :min="128"
                :max="2048"
                style="margin-right: 12px"
                :step="8"
              />
              <NInputNumber
                v-model:value="
                  settings.data.settings.controlnet.detection_resolution
                "
                size="small"
                style="min-width: 96px; width: 96px"
                :min="128"
                :max="2048"
                :step="8"
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
                v-model:value="settings.data.settings.controlnet.seed"
                size="small"
                :min="-1"
                :max="999_999_999_999"
                style="flex-grow: 1"
              />
            </div>

            <!-- Is preprocessed -->
            <div class="flex-container">
              <p class="slider-label">Is Preprocessed</p>
              <NSwitch
                v-model:value="
                  settings.data.settings.controlnet.is_preprocessed
                "
              />
            </div>

            <!-- Save preprocessed -->
            <div class="flex-container">
              <p class="slider-label">Save Preprocessed</p>
              <NSwitch
                v-model:value="
                  settings.data.settings.controlnet.save_preprocessed
                "
              />
            </div>

            <!-- Return preprocessed -->
            <div class="flex-container">
              <p class="slider-label">Return Preprocessed</p>
              <NSwitch
                v-model:value="
                  settings.data.settings.controlnet.return_preprocessed
                "
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
          :current-image="global.state.controlnet.currentImage"
          :images="global.state.controlnet.images"
          :data="settings.data.settings.controlnet"
          @image-clicked="global.state.controlnet.currentImage = $event"
        />

        <OutputStats
          style="margin-top: 12px"
          :gen-data="global.state.controlnet.genData"
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
import BatchSizeInput from "@/components/generate/BatchSizeInput.vue";
import DimensionsInput from "@/components/generate/DimensionsInput.vue";
import SamplerPicker from "@/components/generate/SamplerPicker.vue";
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
  NSwitch,
  NTooltip,
  useMessage,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { computed, onUnmounted } from "vue";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const settings = useSettings();
const messageHandler = useMessage();

const promptCount = computed(() => {
  return settings.data.settings.controlnet.prompt.split(spaceRegex).length - 1;
});
const negativePromptCount = computed(() => {
  return (
    settings.data.settings.controlnet.negative_prompt.split(spaceRegex).length -
    1
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
  settings.data.settings.controlnet.image = base64Image;
};

const generate = () => {
  if (settings.data.settings.controlnet.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }
  global.state.generating = true;

  const seed = checkSeed(settings.data.settings.controlnet.seed);

  fetch(`${serverUrl}/api/generate/controlnet`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        prompt: settings.data.settings.controlnet.prompt,
        image: settings.data.settings.controlnet.image,
        id: uuidv4(),
        negative_prompt: settings.data.settings.controlnet.negative_prompt,
        width: settings.data.settings.controlnet.width,
        height: settings.data.settings.controlnet.height,
        steps: settings.data.settings.controlnet.steps,
        guidance_scale: settings.data.settings.controlnet.cfg_scale,
        seed: seed,
        batch_size: settings.data.settings.controlnet.batch_size,
        batch_count: settings.data.settings.controlnet.batch_count,
        controlnet: settings.data.settings.controlnet.controlnet,
        controlnet_conditioning_scale:
          settings.data.settings.controlnet.controlnet_conditioning_scale,
        detection_resolution:
          settings.data.settings.controlnet.detection_resolution,
        scheduler: settings.data.settings.controlnet.sampler,
        sigmas: settings.data.settings.controlnet.sigmas,
        sampler_settings:
          settings.data.settings.sampler_config[
            settings.data.settings.controlnet.sampler
          ],

        canny_low_threshold: 100,
        canny_high_threshold: 200,
        mlsd_thr_v: 0.1,
        mlsd_thr_d: 0.1,
        is_preprocessed: settings.data.settings.controlnet.is_preprocessed,
        save_preprocessed: settings.data.settings.controlnet.save_preprocessed,
        return_preprocessed:
          settings.data.settings.controlnet.return_preprocessed,
      },
      model: settings.data.settings.model?.name,
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res.json().then((data) => {
        global.state.controlnet.images = data.images;
        global.state.controlnet.currentImage = data.images[0];
        global.state.progress = 0;
        global.state.total_steps = 0;
        global.state.current_step = 0;

        global.state.controlnet.genData = {
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

// Burner clock
const burner = new BurnerClock(
  settings.data.settings.controlnet,
  settings,
  generate
);
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
