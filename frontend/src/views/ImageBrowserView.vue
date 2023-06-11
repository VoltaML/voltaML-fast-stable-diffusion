<template>
  <div
    style="
      width: calc(100vw - 98px);
      height: 48px;
      border-bottom: #505050 1px solid;
      margin-top: 53px;
      display: flex;
      justify-content: end;
      align-items: center;
      padding-right: 24px;
      position: fixed;
      top: 0;
      z-index: 1;
    "
    class="top-bar"
  >
    <NIcon style="margin-right: 12px" size="24">
      <GridOutline />
    </NIcon>
    <NSlider
      :tooltip="false"
      style="width: 20vw"
      :min="100"
      :max="500"
      v-model:value="gridWidth"
    />
  </div>
  <div class="main-container" style="margin-top: 114px">
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
    <NModal
      v-model:show="showImageModal"
      closable
      mask-closable
      preset="card"
      style="width: 85vw"
      title="Image Info"
    >
      <NGrid cols="2" x-gap="12">
        <!-- Left side -->
        <NGi>
          <NImage
            :src="imageSrc"
            object-fit="contain"
            style="width: 100%; height: auto; justify-content: center"
            :img-props="{ style: { width: '40vw', maxHeight: '70vh' } }"
          />
          <NGrid cols="2" x-gap="4" y-gap="4" style="margin-top: 12px">
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
            <NGi span="2">
              <SendOutputTo
                :output="global.state.imageBrowser.currentImageByte64"
                :card="false"
              />
            </NGi>
          </NGrid>
        </NGi>

        <!-- Right side -->
        <NGi>
          <NScrollbar>
            <NDescriptions
              v-if="global.state.imageBrowser.currentImageMetadata.size !== 0"
              :column="2"
              size="large"
            >
              <NDescriptionsItem
                :label="toDescriptionString(key.toString())"
                content-style="max-width: 100px; word-wrap: break-word;"
                style="margin: 4px"
                v-for="(item, key) of global.state.imageBrowser
                  .currentImageMetadata"
                v-bind:key="item.toString()"
              >
                {{ item }}
              </NDescriptionsItem>
            </NDescriptions>
          </NScrollbar>
        </NGi>
      </NGrid>
    </NModal>
    <div class="image-grid-container" ref="scrollComponent">
      <img
        :src="urlFromPath(i.path)"
        v-for="(i, index) in imgData.slice(0, computedImgDataLimit)"
        v-bind:key="index"
        style="width: 100%; height: auto; border-radius: 8px; cursor: pointer"
        @click="imgClick(index)"
      />
    </div>
  </div>
</template>

<script lang="ts" setup>
import SendOutputTo from "@/components/SendOutputTo.vue";
import type { imgData as IImgData } from "@/core/interfaces";
import { serverUrl } from "@/env";
import { Download, GridOutline, TrashBin } from "@vicons/ionicons5";
import {
  NButton,
  NDescriptions,
  NDescriptionsItem,
  NGi,
  NGrid,
  NIcon,
  NImage,
  NModal,
  NScrollbar,
  NSlider,
} from "naive-ui";
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import { useSettings } from "../store/settings";
import { useState } from "../store/state";

const global = useState();
const conf = useSettings();
const showDeleteModal = ref(false);

const showImageModal = ref(false);

const scrollComponent = ref<HTMLElement | null>(null);
const imageLimit = ref(30);

const gridWidth = ref(250);
const computedGridWidth = computed(() => {
  return gridWidth.value.toString() + "px";
});

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
      // Close the modal
      showImageModal.value = false;

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

      // Remove the image from the list
      imgData.splice(
        imgData.findIndex(
          (i) => i.path === global.state.imageBrowser.currentImage.path
        ),
        1
      );
    });
}

function toDescriptionString(str: string): string {
  const upper = str.charAt(0).toUpperCase() + str.slice(1);
  return upper.replace(/_/g, " ");
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

function imgClick(i: number) {
  global.state.imageBrowser.currentImage = imgData[i];
  setByte64FromImage(imgData[i].path);
  const url = new URL(`${serverUrl}/api/output/data/`);
  url.searchParams.append("filename", imgData[i].path);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = data;
    });
  showImageModal.value = true;
}

const imgData: IImgData[] = reactive([]);
const computedImgDataLimit = computed(() => {
  return Math.min(imgData.length, imageLimit.value);
});

async function refreshImages() {
  // Clear the data
  imgData.splice(0, imgData.length);

  // Fetch new data
  await fetch(`${serverUrl}/api/output/txt2img`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: IImgData) => {
        imgData.push(item);
      });
    });

  await fetch(`${serverUrl}/api/output/img2img`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: IImgData) => {
        imgData.push(item);
      });
    });

  await fetch(`${serverUrl}/api/output/extra`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: IImgData) => {
        imgData.push(item);
      });
    });

  imgData.sort((a, b) => {
    return b.time - a.time;
  });
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const handleScroll = (e: Event) => {
  let element = scrollComponent.value;
  if (element === null) {
    return;
  }
  if (element.getBoundingClientRect().bottom - 200 < window.innerHeight) {
    imageLimit.value += 30;
  }
};

onMounted(() => {
  window.addEventListener("scroll", handleScroll);
});

onUnmounted(() => {
  window.removeEventListener("scroll", handleScroll);
});

refreshImages();

const backgroundColor = computed(() => {
  if (conf.data.settings.frontend.theme === "dark") {
    return "#121215";
  } else {
    return "#fff";
  }
});
</script>

<style scoped>
.img-slider {
  aspect-ratio: 1/1;
  height: 182px;
  width: auto;
}

.image-grid-container {
  display: grid;
  gap: 6px;
  grid-template-columns: repeat(
    auto-fit,
    minmax(v-bind(computedGridWidth), 1fr)
  );
  grid-auto-flow: dense;
}

.top-bar {
  background-color: v-bind(backgroundColor);
}
</style>
