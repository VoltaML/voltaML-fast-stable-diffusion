<template>
  <div class="main-container">
    <!-- Progress bar -->
    <div class="progress-container">
      <NProgress
        type="line"
        :percentage="60"
        indicator-placement="outside"
        processing
        color="#63e2b7"
      />
    </div>

    <!-- Main -->

    <NGrid cols="2" x-gap="12">
      <NGi>
        <NSpace vertical class="left-container">
          <!-- Backend selection -->
          <NSpace>
            <p>Backend</p>
            <NSpace align="center" style="height: 100%">
              <NRadioGroup
                v-model:value="selectedBackend"
                name="backend-selection"
              >
                <NRadioButton
                  v-for="backend in [
                    { label: 'PyTorch', value: 'PyTorch' },
                    { label: 'TensorFlow', value: 'TensorFlow' },
                  ]"
                  :key="backend.value"
                  :value="backend.value"
                  :label="backend.label"
                />
              </NRadioGroup>
            </NSpace>
          </NSpace>

          <!-- Prompt -->
          <NInput v-model:value="prompt" type="textarea" placeholder="Prompt" />
          <NInput
            v-model:value="negativePrompt"
            type="textarea"
            placeholder="Negative prompt"
          />

          <!-- Sampler -->
          <NSpace>
            <p>Sampler</p>
            <NSpace align="center" style="height: 100%">
              <NRadioGroup
                v-model:value="selectedSampler"
                name="radiobuttongroup1"
              >
                <NRadioButton
                  v-for="backend in [
                    { label: 'Euler A', value: 'k_euler_a' },
                    { label: 'DDIM', value: 'ddim' },
                    { label: 'DPM', value: 'dpm' },
                    { label: 'LMS', value: 'lms' },
                  ]"
                  :key="backend.value"
                  :value="backend.value"
                  :label="backend.label"
                />
              </NRadioGroup>
            </NSpace>
          </NSpace>

          <!-- Dimensions -->

          <div class="flex-container">
            <p class="slider-label">Width</p>
            <NSlider
              v-model:value="width"
              :min="128"
              :max="2048"
              :step="16"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="width"
              size="small"
              style="width: 128px"
            />
          </div>
          <div class="flex-container">
            <p class="slider-label">Height</p>
            <NSlider
              v-model:value="height"
              :min="128"
              :max="2048"
              :step="16"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="height"
              size="small"
              style="width: 128px"
            />
          </div>

          <!-- Steps -->
          <div class="flex-container">
            <p class="slider-label">Steps</p>
            <NSlider
              v-model:value="steps"
              :min="5"
              :max="300"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="steps"
              size="small"
              style="width: 128px"
              :min="5"
              :max="300"
            />
          </div>

          <!-- CFG Scale -->
          <div class="flex-container">
            <p class="slider-label">CFG Scale</p>
            <NSlider
              v-model:value="cfgScale"
              :min="1"
              :max="30"
              :step="0.5"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="cfgScale"
              size="small"
              style="width: 128px"
              :min="1"
              :max="30"
              :step="0.5"
            />
          </div>

          <!-- Seed -->
          <div class="flex-container">
            <p class="slider-label">Seed</p>
            <NInputNumber
              v-model:value="seed"
              size="small"
              :min="-1"
              :max="999999999"
              style="width: 100%"
            />
          </div>

          <!-- Number of images -->
          <div class="flex-container">
            <p class="slider-label">Batch size</p>
            <NSlider
              v-model:value="steps"
              :min="5"
              :max="300"
              style="margin-right: 12px"
            />
            <NInputNumber
              v-model:value="steps"
              size="small"
              style="width: 128px"
              :min="5"
              :max="300"
            />
          </div>

          <!-- Generate button -->
          <NSpace justify="center">
            <NButton type="success">Generate</NButton>
          </NSpace>
        </NSpace>
      </NGi>

      <!-- Split -->

      <!-- Images -->
      <NGi>
        <NSpace justify="center">
          <NImage
            src="https://github.com/VoltaML/voltaML-fast-stable-diffusion/blob/main/static/0.png?raw=true"
          />
        </NSpace>
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import {
  NButton,
  NGi,
  NGrid,
  NImage,
  NInput,
  NInputNumber,
  NProgress,
  NRadioButton,
  NRadioGroup,
  NSlider,
  NSpace,
} from "naive-ui";
import { ref } from "vue";

const width = ref(512);
const height = ref(512);
const seed = ref(-1);
const steps = ref(50);
const cfgScale = ref(7);
const selectedSampler = ref("k_euler_a");
const selectedBackend = ref("PyTorch");
const prompt = ref("");
const negativePrompt = ref("");
</script>

<style scoped>
.left-container {
  margin: 0 12px;
}

.progress-container {
  margin: 12px;
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
}

.main-container {
  margin: 18px;
}
</style>
