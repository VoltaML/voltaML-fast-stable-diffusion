<template>
  <div style="margin: 18px">
    <NModal
      v-model:show="showDeleteModal"
      :mask-closable="false"
      preset="confirm"
      type="error"
      title="Delete Image"
      content="Do you want to delete this image? This action cannot be undone."
      positive-text="Confirm"
      negative-text="Cancel"
      transform-origin="center"
      @positive-click="deleteImage"
      @negative-click="showDeleteModal = false"
    />
    <NGrid cols="1 850:3" x-gap="12px">
      <!-- Top -->
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
          style="
            width: 100%;
            height: 100%;
            justify-content: center;
            margin: 8px;
          "
          :img-props="{ style: { maxWidth: '95%', maxHeight: '95%' } }"
        />
      </NGi>
      <NGi span="1">
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

              <NTabPane
                name="Img2Img"
                style="height: calc(((100vh - 200px) - 53px) - 24px)"
              >
                <NScrollbar trigger="hover" style="height: 100%">
                  <span
                    v-for="(i, index) in img2imgData"
                    @click="img2imgClick(index)"
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

              <NTabPane
                name="Extra"
                style="height: calc(((100vh - 200px) - 53px) - 24px)"
              >
                <NScrollbar trigger="hover" style="height: 100%">
                  <span
                    v-for="(i, index) in extraData"
                    @click="extraClick(index)"
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
            </NTabs>
          </NCard>
        </div>
      </NGi>

      <!-- Middle -->
      <NGi span="3">
        <NGrid
          cols="2"
          x-gap="12"
          v-if="global.state.imageBrowser.currentImage.path !== ''"
        >
          <NGi>
            <SendOutputTo
              :output="global.state.imageBrowser.currentImageByte64"
            />
          </NGi>
          <NGi>
            <NCard style="margin: 12px 0" title="Manage">
              <NGrid cols="4" x-gap="4" y-gap="4">
                <NGi>
                  <NButton
                    type="success"
                    @click="downloadImage"
                    style="width: 100%"
                    ghost
                    ><template #icon>
                      <NIcon>
                        <Download />
                      </NIcon> </template
                    >Download</NButton
                  >
                </NGi>

                <NGi>
                  <NButton
                    type="error"
                    @click="showDeleteModal = true"
                    style="width: 100%"
                    ghost
                  >
                    <template #icon>
                      <NIcon>
                        <TrashBin />
                      </NIcon>
                    </template>
                    Delete</NButton
                  >
                </NGi>
              </NGrid>
            </NCard>
          </NGi>
        </NGrid>
      </NGi>

      <!-- Bottom -->
      <NGi span="3">
        <NDescriptions
          bordered
          v-if="global.state.imageBrowser.currentImageMetadata.size !== 0"
        >
          <NDescriptionsItem
            :label="key.toString()"
            content-style="max-width: 100px"
            v-for="(item, key) of global.state.imageBrowser
              .currentImageMetadata"
            v-bind:key="item.toString()"
          >
            {{ item }}
          </NDescriptionsItem>
        </NDescriptions>
      </NGi>
    </NGrid>
  </div>
</template>

<script lang="ts" setup>
import SendOutputTo from "@/components/SendOutputTo.vue";
import type { imgData } from "@/core/interfaces";
import { serverUrl } from "@/env";
import { Download, TrashBin } from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NDescriptions,
  NDescriptionsItem,
  NGi,
  NGrid,
  NIcon,
  NImage,
  NModal,
  NScrollbar,
  NTabPane,
  NTabs,
} from "naive-ui";
import { computed, reactive, ref } from "vue";
import { useState } from "../store/state";

const global = useState();
const showDeleteModal = ref(false);

function urlFromPath(path: string) {
  const url = new URL(path, serverUrl);
  return url.href;
}

const imageSrc = computed(() => {
  const url = urlFromPath(global.state.imageBrowser.currentImage.path);
  return url;
});

function deleteImage() {
  const url = new URL(`${serverUrl}/api/output/delete/`);
  url.searchParams.append(
    "filename",
    global.state.imageBrowser.currentImage.path
  );
  fetch(url, { method: "DELETE" })
    .then((res) => res.json())
    .then(() => {
      global.state.imageBrowser.currentImage = {
        path: "",
        id: "",
        time: 0,
      };
      global.state.imageBrowser.currentImageByte64 = "";
      global.state.imageBrowser.currentImageMetadata = new Map<
        string,
        string
      >();

      // Refresh the image list
      refreshImages();
    });
}

function downloadImage() {
  const url = urlFromPath(global.state.imageBrowser.currentImage.path);

  fetch(url)
    .then((res) => res.blob())
    .then((blob) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = function () {
        const base64data = reader.result;
        if (base64data !== null) {
          // Download the image
          const a = document.createElement("a");
          a.href = base64data.toString();
          a.download = global.state.imageBrowser.currentImage.id;
          a.target = "_blank";
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        } else {
          console.log("base64data is null!");
        }
      };
    });
}

function setByte64FromImage(path: string) {
  const url = urlFromPath(path);
  fetch(url)
    .then((res) => res.blob())
    .then((blob) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = function () {
        const base64data = reader.result;
        if (base64data !== null) {
          global.state.imageBrowser.currentImageByte64 = base64data.toString();
        } else {
          console.log("base64data is null!");
        }
      };
    });
}

function txt2imgClick(i: number) {
  global.state.imageBrowser.currentImage = txt2imgData[i];
  setByte64FromImage(txt2imgData[i].path);
  const url = new URL(`${serverUrl}/api/output/data/`);
  url.searchParams.append("filename", txt2imgData[i].path);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = data;
    });
}

function img2imgClick(i: number) {
  global.state.imageBrowser.currentImage = img2imgData[i];
  setByte64FromImage(img2imgData[i].path);
  const url = new URL(`${serverUrl}/api/output/data/`);
  url.searchParams.append("filename", img2imgData[i].path);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = data;
    });
}

function extraClick(i: number) {
  global.state.imageBrowser.currentImage = extraData[i];
  setByte64FromImage(extraData[i].path);
  const url = new URL(`${serverUrl}/api/output/data/`);
  url.searchParams.append("filename", extraData[i].path);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = data;
    });
}

const txt2imgData: imgData[] = reactive([]);
const img2imgData: imgData[] = reactive([]);
const extraData: imgData[] = reactive([]);

function refreshImages() {
  // Clear the data
  txt2imgData.splice(0, txt2imgData.length);
  img2imgData.splice(0, img2imgData.length);
  extraData.splice(0, extraData.length);

  // Fetch new data
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

  fetch(`${serverUrl}/api/output/img2img`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: imgData) => {
        img2imgData.push(item);
      });
      img2imgData.sort((a, b) => {
        return b.time - a.time;
      });
    });

  fetch(`${serverUrl}/api/output/extra`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: imgData) => {
        extraData.push(item);
      });
      extraData.sort((a, b) => {
        return b.time - a.time;
      });
    });
}

refreshImages();
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
