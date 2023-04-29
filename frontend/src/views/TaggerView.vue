<template>
  <div class="main-container">
    <!-- Main -->
    <NGrid cols="1 m:2" x-gap="12" responsive="screen">
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

        <NCard>
          <div class="flex-container">
            <p class="slider-label">Weighted</p>
            <NSwitch v-model:value="weighted" />
          </div>

          <NInput
            v-model:value="computedPrompt"
            type="textarea"
            placeholder="Prompt"
            show-count
          >
            <template #count>{{ promptCount }}</template>
          </NInput>
          <NInput
            v-model:value="computedNegativePrompt"
            type="textarea"
            placeholder="Negative prompt"
            show-count
          >
            <template #count>{{ negativePromptCount }}</template>
          </NInput>
        </NCard>
      </NGi>
    </NGrid>
  </div>
</template>

<script setup lang="ts">
import "@/assets/2img.css";
import GenerateSection from "@/components/GenerateSection.vue";
import ImageUpload from "@/components/ImageUpload.vue";
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
import { computed, ref } from "vue";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";

const global = useState();
const conf = useSettings();
const messageHandler = useMessage();

const imageSelectCallback = (base64Image: string) => {
  conf.data.settings.tagger.image = base64Image;
};

const generate = () => {
  global.state.generating = true;
  fetch(`${serverUrl}/api/generate/interrogate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
        image: conf.data.settings.tagger.image,
        id: uuidv4(),
        strength: conf.data.settings.tagger.treshold,
      },
      model: conf.data.settings.tagger.model,
    }),
  })
    .then((res) => {
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      global.state.generating = false;
      res
        .json()
        .then(
          (data: {
            positive: Map<string, number>;
            negative: Map<string, number>;
          }) => {
            global.state.tagger.positivePrompt = data.positive;
            global.state.tagger.negativePrompt = data.negative;
            console.log(data);
          }
        );
    })
    .catch((err) => {
      global.state.generating = false;
      messageHandler.error(err);
      console.log(err);
    });
};

const weighted = ref(false);

function MapToPrompt(map: Map<string, number>) {
  if (weighted.value) {
    // Convert to weighted prompt: (token: weight)
    let weightedPrompt = Array<string>();
    for (const [key, value] of map) {
      if (value.toFixed(2) === "1.00") {
        weightedPrompt.push(`${key}`);
        continue;
      } else {
        weightedPrompt.push(`(${key}: ${value.toFixed(2)})`);
      }
    }
    return weightedPrompt.join(", ");
  } else {
    // Append as a string but sorted according to weight
    let prompt = Array<string>();
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for (const [key, value] of map) {
      prompt.push(key);
    }
    return prompt.join(", ");
  }
}

const computedPrompt = computed(() => {
  const sortedMap = new Map(
    [...global.state.tagger.positivePrompt].sort((a, b) => b[1] - a[1])
  );
  return MapToPrompt(sortedMap);
});

const computedNegativePrompt = computed(() => {
  const sortedMap = new Map(
    [...global.state.tagger.negativePrompt].sort((a, b) => b[1] - a[1])
  );
  return MapToPrompt(sortedMap);
});

const promptCount = computed(() => {
  return computedPrompt.value.split(spaceRegex).length - 1;
});
const negativePromptCount = computed(() => {
  return computedNegativePrompt.value.split(spaceRegex).length - 1;
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
