<template>
  <NGrid cols="1 850:3">
    <NGi
      span="2"
      style="
        height: calc((100vh - 170px) - 24px);
        display: inline-flex;
        justify-content: center;
      "
    >
      <NImage
        v-if="global.state.imageBrowser.currentImage.path !== ''"
        :src="imageSrc"
        object-fit="contain"
        style="width: 100%; height: 100%; justify-content: center; margin: 8px"
        :img-props="{ style: { maxWidth: '95%', maxHeight: '95%' } }"
      />
    </NGi>
    <NGi>
      <div style="height: 100%; width: 100%">
        <NCard>
          <NTabs type="segment" style="height: 100%">
            <NTabPane
              name="Txt2Img"
              style="height: calc(((100vh - 200px) - 53px) - 24px)"
            >
              <NScrollbar trigger="hover" style="height: 100%">
                <span
                  v-for="(i, index) in txt2imgData"
                  @click="txt2imgClick(index)"
                  v-bind:key="index"
                  class="img-container"
                >
                  <NImage
                    class="img-slider"
                    :src="urlFromPath(i.path)"
                    lazy
                    preview-disabled
                    style="justify-content: center"
                  />
                </span>
              </NScrollbar>
            </NTabPane>
            <NTabPane name="Img2Img"> </NTabPane>
          </NTabs>
        </NCard>
      </div>
    </NGi>
    <NGi span="3">
      <NDescriptions bordered>
        <NDescriptionsItem label="File" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImage.path.split("/").pop() }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Model" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImageMetadata.model }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Seed" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImageMetadata.seed }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Prompt" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImageMetadata.prompt }}
        </NDescriptionsItem>
        <NDescriptionsItem
          label="Negative Prompt"
          content-style="max-width: 100px"
        >
          {{ global.state.imageBrowser.currentImageMetadata.negative_prompt }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Steps" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImageMetadata.steps }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Width" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImageMetadata.width }}
        </NDescriptionsItem>
        <NDescriptionsItem label="Height" content-style="max-width: 100px">
          {{ global.state.imageBrowser.currentImageMetadata.height }}
        </NDescriptionsItem>
        <NDescriptionsItem
          label="Guidance Scale"
          content-style="max-width: 100px"
        >
          {{ global.state.imageBrowser.currentImageMetadata.guidance_scale }}
        </NDescriptionsItem>
      </NDescriptions>
    </NGi>
  </NGrid>
</template>

<script lang="ts" setup>
import type { imgData } from "@/core/interfaces";
import { serverUrl } from "@/env";
import {
  NCard,
  NDescriptions,
  NDescriptionsItem,
  NGi,
  NGrid,
  NImage,
  NScrollbar,
  NTabPane,
  NTabs,
} from "naive-ui";
import { computed, reactive } from "vue";
import { useState } from "../store/state";

const global = useState();

function urlFromPath(path: string) {
  const url = new URL(path, serverUrl);
  return url.href;
}

const imageSrc = computed(() => {
  const url = urlFromPath(global.state.imageBrowser.currentImage.path);
  return url;
});

function txt2imgClick(i: number) {
  global.state.imageBrowser.currentImage = txt2imgData[i];
  console.log(txt2imgData[i].path);
  const url = new URL(`${serverUrl}/api/output/data/`);
  url.searchParams.append("filename", txt2imgData[i].path);
  console.log(url);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = data;
    });
}

const txt2imgData: imgData[] = reactive([]);
fetch(`${serverUrl}/api/output/txt2img`)
  .then((res) => res.json())
  .then((data) => {
    data.forEach((item: imgData) => {
      txt2imgData.push(item);
    });
    txt2imgData.sort((a, b) => {
      return b.time - a.time;
    });
  });
</script>

<style scoped>
.img-slider {
  aspect-ratio: 1/1;
  height: 182px;
  width: auto;
}

.img-container:not(:last-child) {
  margin-right: 6px;
}

.img-container {
  cursor: pointer;
}
</style>
