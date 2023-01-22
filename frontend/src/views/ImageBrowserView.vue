<template>
  <NCard class="main-card">
    <NGrid cols="1 850:3" style="height: calc(94vh - 200px - 64px)">
      <NGi span="2">
        <NCard style="width: 100%; height: 100%; display: inline-flex">
          <NImage
            v-if="global.state.imageBrowser.currentImage.path !== ''"
            class="img-container"
            :src="imageSrc"
            object-fit="contain"
            style="
              width: 100%;
              height: 100%;
              justify-content: center;
              max-height: 61vh;
            "
          />
        </NCard>
      </NGi>
      <NGi>
        <NCard style="height: 100%">
          <NCard title="Prompt" size="small">{{
            global.state.imageBrowser.currentImageMetadata.prompt
          }}</NCard>
          <NCard title="Negative Prompt" size="small">{{
            global.state.imageBrowser.currentImageMetadata.negative_prompt
          }}</NCard>
          <NGrid cols="2">
            <NGi>
              <NCard title="Width" size="small">{{
                global.state.imageBrowser.currentImageMetadata.width
              }}</NCard>
            </NGi>
            <NGi>
              <NCard title="Height" size="small">{{
                global.state.imageBrowser.currentImageMetadata.height
              }}</NCard>
            </NGi>
          </NGrid>
          <NGrid cols="2">
            <NGi>
              <NCard title="Steps" size="small">{{
                global.state.imageBrowser.currentImageMetadata.steps
              }}</NCard>
            </NGi>
            <NGi>
              <NCard title="Guidance Scale" size="small">{{
                global.state.imageBrowser.currentImageMetadata.guidance_scale
              }}</NCard>
            </NGi>
          </NGrid>
          <NGrid cols="2">
            <NGi>
              <NCard title="Seed" size="small">{{
                global.state.imageBrowser.currentImageMetadata.seed
              }}</NCard>
            </NGi>
            <NGi>
              <NCard title="Model" size="small">{{
                global.state.imageBrowser.currentImageMetadata.model
              }}</NCard>
            </NGi>
          </NGrid>
        </NCard>
      </NGi>
    </NGrid>
    <NCard style="height: 200px">
      <NTabs type="segment">
        <NTabPane name="Txt2Img">
          <NScrollbar x-scrollable trigger="none">
            <div style="white-space: nowrap">
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
            </div>
          </NScrollbar>
        </NTabPane>
        <NTabPane name="Img2Img"> </NTabPane>
      </NTabs>
    </NCard>
  </NCard>
</template>

<script lang="ts" setup>
import type { imgData } from "@/core/interfaces";
import { serverUrl } from "@/env";
import {
  NCard,
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
  });
</script>

<style scoped>
.main-card {
  height: 94vh;
}
.img-slider {
  aspect-ratio: 1/1;
  height: 100px;
  width: auto;
}

.img-container:not(:last-child) {
  margin-right: 8px;
}

.img-container {
  cursor: pointer;
}
</style>
