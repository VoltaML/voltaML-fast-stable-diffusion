<template>
  <div class="main-container">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
      <NGi>
        <NCard title="Settings">
          <NSpace vertical class="left-container">
            <!-- Prompt -->
            <Prompt tab="txt2img" />

            <!-- Sampler -->
            <SamplerPicker :type="'txt2img'" />

            <!-- Dimensions -->
            <DimensionsInput
              :dimensions-object="settings.data.settings.txt2img"
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
                v-model:value="settings.data.settings.txt2img.steps"
                :min="5"
                :max="300"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.txt2img.steps"
                size="small"
                style="min-width: 96px; width: 96px"
              />
            </div>

            <CFGScale tab="txt2img" />

            <SAGInput tab="txt2img" />

            <!-- Number of images -->
            <div class="flex-container">
              <NTooltip style="max-width: 600px">
                <template #trigger>
                  <p class="slider-label">Batch Count</p>
                </template>
                Number of images to generate after each other.
              </NTooltip>
              <NSlider
                v-model:value="settings.data.settings.txt2img.batch_count"
                :min="1"
                :max="9"
                style="margin-right: 12px"
              />
              <NInputNumber
                v-model:value="settings.data.settings.txt2img.batch_count"
                size="small"
                style="min-width: 96px; width: 96px"
              />
            </div>

            <BatchSizeInput
              :batch-size-object="settings.data.settings.txt2img"
            />

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
                v-model:value="settings.data.settings.txt2img.seed"
                size="small"
                style="flex-grow: 1"
              />
            </div>
          </NSpace>
        </NCard>

        <ResizeFromDimensionsInput
          :dimensions-object="settings.data.settings.txt2img"
          v-if="settings.data.settings.model?.type === 'SDXL'"
        />
        <XLRefiner v-if="isSelectedModelSDXL" />
        <HighResFix tab="txt2img" />
        <Upscale tab="txt2img" />
      </NGi>

      <!-- Split -->

      <!-- Images -->
      <NGi>
        <GenerateSection :generate="generate" />

        <ImageOutput
          :current-image="global.state.txt2img.currentImage"
          :images="global.state.txt2img.images"
          :data="settings.data.settings.txt2img"
          @image-clicked="global.state.txt2img.currentImage = $event"
        />

        <OutputStats
          style="margin-top: 12px"
          :gen-data="global.state.txt2img.genData"
        />
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import {
  BatchSizeInput,
  CFGScale,
  DimensionsInput,
  GenerateSection,
  HighResFix,
  ImageOutput,
  OutputStats,
  Prompt,
  ResizeFromDimensionsInput,
  SAGInput,
  SamplerPicker,
  Upscale,
  XLRefiner,
} from "@/components";
import { serverUrl } from "@/env";
import {
  NCard,
  NGi,
  NGrid,
  NInputNumber,
  NSlider,
  NSpace,
  NTooltip,
  useMessage,
} from "naive-ui";
import { v4 as uuidv4 } from "uuid";
import { computed, onUnmounted } from "vue";
import { BurnerClock } from "../../clock";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const settings = useSettings();
const messageHandler = useMessage();

const isSelectedModelSDXL = computed(() => {
  return settings.data.settings.model?.type === "SDXL";
});

const checkSeed = (seed: number) => {
  // If -1 create random seed
  if (seed === -1) {
    seed = Math.floor(Math.random() * 999_999_999_999);
  }

  return seed;
};

const generate = () => {
  if (settings.data.settings.txt2img.seed === null) {
    messageHandler.error("Please set a seed");
    return;
  }
  global.state.generating = true;

  const seed = checkSeed(settings.data.settings.txt2img.seed);

  fetch(`${serverUrl}/api/generate/txt2img`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        id: uuidv4(),
        prompt: settings.data.settings.txt2img.prompt,
        negative_prompt: settings.data.settings.txt2img.negative_prompt,
        width: settings.data.settings.txt2img.width,
        height: settings.data.settings.txt2img.height,
        steps: settings.data.settings.txt2img.steps,
        guidance_scale: settings.data.settings.txt2img.cfg_scale,
        seed: seed,
        batch_size: settings.data.settings.txt2img.batch_size,
        batch_count: settings.data.settings.txt2img.batch_count,
        scheduler: settings.data.settings.txt2img.sampler,
        self_attention_scale:
          settings.data.settings.txt2img.self_attention_scale,
        sigmas: settings.data.settings.txt2img.sigmas,
        sampler_settings:
          settings.data.settings.sampler_config[
            settings.data.settings.txt2img.sampler
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
      backend: "PyTorch",
      autoload: false,
      flags: {
        ...(isSelectedModelSDXL.value && global.state.txt2img.sdxl_resize
          ? {
              sdxl: {
                original_size: {
                  width: settings.data.settings.flags.sdxl.original_size.width,
                  height:
                    settings.data.settings.flags.sdxl.original_size.height,
                },
              },
            }
          : {}),
        ...(settings.data.settings.txt2img.highres.enabled
          ? {
              highres_fix: {
                mode: settings.data.settings.txt2img.highres.mode,
                image_upscaler:
                  settings.data.settings.txt2img.highres.image_upscaler,
                scale: settings.data.settings.txt2img.highres.scale,
                latent_scale_mode:
                  settings.data.settings.txt2img.highres.latent_scale_mode,
                strength: settings.data.settings.txt2img.highres.strength,
                steps: settings.data.settings.txt2img.highres.steps,
                antialiased: settings.data.settings.txt2img.highres.antialiased,
              },
            }
          : global.state.txt2img.refiner
          ? {
              refiner: {
                model: settings.data.settings.flags.refiner.model,
                aesthetic_score:
                  settings.data.settings.flags.refiner.aesthetic_score,
                negative_aesthetic_score:
                  settings.data.settings.flags.refiner.negative_aesthetic_score,
                steps: settings.data.settings.flags.refiner.steps,
                strength: settings.data.settings.flags.refiner.strength,
              },
            }
          : {}),
        ...(settings.data.settings.txt2img.upscale.enabled
          ? {
              upscale: {
                upscale_factor:
                  settings.data.settings.txt2img.upscale.upscale_factor,
                tile_size: settings.data.settings.txt2img.upscale.tile_size,
                tile_padding:
                  settings.data.settings.txt2img.upscale.tile_padding,
                model: settings.data.settings.txt2img.upscale.model,
              },
            }
          : {}),
      },
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res.json().then((data) => {
        global.state.txt2img.images = data.images;
        global.state.txt2img.currentImage = data.images[0];
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
    });
};

// Burner clock
const burner = new BurnerClock(
  settings.data.settings.txt2img,
  settings,
  generate
);
onUnmounted(() => {
  burner.cleanup();
});
</script>
