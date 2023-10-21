<template>
  <div>
    <div
      style="
        width: calc(100vw - 98px);
        height: 48px;
        border-bottom: #505050 1px solid;
        margin-top: 52px;
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
      <NInput
        v-model:value="itemFilter"
        style="margin: 0 12px"
        placeholder="Filter"
      />
      <NIcon style="margin-right: 12px" size="22">
        <GridOutline />
      </NIcon>
      <NSlider
        style="width: 50vw"
        :min="1"
        :max="10"
        v-model:value="settings.data.settings.frontend.image_browser_columns"
      >
      </NSlider>
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
        close-on-esc
        preset="card"
        style="width: 85vw"
        title="Image Info"
        id="image-modal"
      >
        <NGrid cols="1 m:2" x-gap="12" y-gap="12" responsive="screen">
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
                  :data="global.state.imageBrowser.currentImageMetadata"
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
                  :label="convertToTextString(key.toString())"
                  content-style="max-width: 100px; word-wrap: break-word;"
                  style="margin: 4px"
                  v-for="(item, key) of global.state.imageBrowser
                    .currentImageMetadata"
                  v-bind:key="item.toString()"
                >
                  {{
                    key.toString() === "scheduler"
                      ? getNamedSampler(item.toString())
                      : item
                  }}
                </NDescriptionsItem>
              </NDescriptions>
            </NScrollbar>
          </NGi>
        </NGrid>
      </NModal>
      <div ref="scrollComponent">
        <div class="image-grid">
          <div
            v-for="(column, column_index) in columns"
            v-bind:key="column_index"
            class="image-column"
            ref="gridColumnRefs"
          >
            <img
              v-for="(item, item_index) in column"
              :src="urlFromPath(item.path)"
              v-bind:key="item_index"
              style="
                width: 100%;
                height: auto;
                border-radius: 8px;
                cursor: pointer;
                margin-bottom: 6px;
              "
              @click="imgClick(column_index, item_index)"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import SendOutputTo from "@/components/SendOutputTo.vue";
import type { imgData as IImgData } from "@/core/interfaces";
import { serverUrl } from "@/env";
import { convertToTextString, urlFromPath } from "@/functions";
import { themeOverridesKey } from "@/injectionKeys";
import { Download, GridOutline, TrashBin } from "@vicons/ionicons5";
import {
  NButton,
  NDescriptions,
  NDescriptionsItem,
  NGi,
  NGrid,
  NIcon,
  NImage,
  NInput,
  NModal,
  NScrollbar,
  NSlider,
} from "naive-ui";
import { computed, inject, onMounted, onUnmounted, reactive, ref } from "vue";
import { diffusersSchedulerTuple, useSettings } from "../store/settings";
import { useState } from "../store/state";

const global = useState();
const settings = useSettings();
const theme = inject(themeOverridesKey);
const showDeleteModal = ref(false);

const showImageModal = ref(false);

const scrollComponent = ref<HTMLElement | null>(null);
const imageLimit = ref(30);
const itemFilter = ref("");

const gridColumnRefs = ref<HTMLElement[]>([]);

const imageSrc = computed(() => {
  const url = urlFromPath(global.state.imageBrowser.currentImage.path);
  return url;
});

function deleteImage() {
  const url = new URL(`${serverUrl}/api/outputs/delete/`);
  url.searchParams.append(
    "filename",
    global.state.imageBrowser.currentImage.path
  );
  fetch(url, { method: "DELETE" })
    .then((res) => res.json())
    .then(() => {
      // Close the modal
      showImageModal.value = false;

      // Remove the image from the list
      const index = imgData.findIndex((el) => {
        return el.path === global.state.imageBrowser.currentImage.path;
      });
      imgData.splice(index, 1);

      global.state.imageBrowser.currentImage = {
        path: "",
        id: "",
        time: 0,
      };
      global.state.imageBrowser.currentImageByte64 = "";
      global.state.imageBrowser.currentImageMetadata = {};
    });
}

function downloadImage() {
  const url = urlFromPath(global.state.imageBrowser.currentImage.path);

  fetch(url, { mode: "no-cors" })
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

const currentColumn = ref(0);
const currentRowIndex = ref(0);

function parseMetadataFromString(key: string, value: string) {
  value = value.trim().toLowerCase();

  if (value === "true") {
    return true;
  } else if (value === "false") {
    return false;
  } else {
    if (isFinite(+value)) {
      return +value;
    } else {
      return value;
    }
  }
}

function imgClick(column_index: number, item_index: number) {
  currentRowIndex.value = item_index;
  currentColumn.value = column_index;
  const item = columns.value[column_index][item_index];
  global.state.imageBrowser.currentImage = item;
  setByte64FromImage(item.path);
  const url = new URL(`${serverUrl}/api/outputs/data/`);
  url.searchParams.append("filename", item.path);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = JSON.parse(
        JSON.stringify(data),
        (key, value) => {
          if (typeof value === "string") {
            return parseMetadataFromString(key, value);
          }
          return value;
        }
      );
    });
  showImageModal.value = true;
}

const imgData: IImgData[] = reactive<IImgData[]>([]);

const filteredImgData = computed(() => {
  return imgData.filter((item) => {
    if (itemFilter.value === "") {
      return true;
    }
    return item.path.includes(itemFilter.value);
  });
});

const computedImgDataLimit = computed(() => {
  return Math.min(filteredImgData.value.length, imageLimit.value);
});

const columns = computed(() => {
  const cols: IImgData[][] = [];
  for (
    let i = 0;
    i < settings.data.settings.frontend.image_browser_columns;
    i++
  ) {
    cols.push([]);
  }
  for (let i = 0; i < computedImgDataLimit.value; i++) {
    cols[i % settings.data.settings.frontend.image_browser_columns].push(
      filteredImgData.value[i]
    );
  }
  return cols;
});

async function refreshImages() {
  // Clear the data
  imgData.splice(0, imgData.length);

  // Fetch new data
  await fetch(`${serverUrl}/api/outputs/txt2img`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: IImgData) => {
        imgData.push(item);
      });
    });

  await fetch(`${serverUrl}/api/outputs/img2img`)
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item: IImgData) => {
        imgData.push(item);
      });
    });

  await fetch(`${serverUrl}/api/outputs/extra`)
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
  // Check for the scroll component, as it is required to continue
  let element = scrollComponent.value;
  if (element === null) {
    return;
  }

  // Get the smallest possible value of the bottom of the images columns
  let minBox = 0;
  for (const col of gridColumnRefs.value) {
    const lastImg = col.childNodes.item(
      col.childNodes.length - 2
    ) as HTMLElement;
    const bottombbox = lastImg.getBoundingClientRect().bottom;
    if (minBox === 0) {
      minBox = bottombbox;
    } else if (bottombbox < minBox) {
      minBox = bottombbox;
    }
  }

  // Extend the image limit if the bottom of the images is less than 50px from the bottom of the screen
  if (minBox - 50 < window.innerHeight) {
    if (imageLimit.value >= filteredImgData.value.length) {
      return;
    }
    imageLimit.value += 30;
  }
};

function moveImage(direction: number) {
  const numColumns = settings.data.settings.frontend.image_browser_columns;

  if (direction === -1) {
    // Traverse all the columns before removing one from the currentIndexOfColumn
    if (currentColumn.value > 0) {
      imgClick(currentColumn.value - 1, currentRowIndex.value);
    } else {
      imgClick(numColumns - 1, currentRowIndex.value - 1);
    }
  } else if (direction === 1) {
    // Traverse all the columns before adding one from the currentIndexOfColumn
    if (currentColumn.value < numColumns - 1) {
      imgClick(currentColumn.value + 1, currentRowIndex.value);
    } else {
      imgClick(0, currentRowIndex.value + 1);
    }
  }
}

onMounted(() => {
  // Infinite Scroll
  window.addEventListener("scroll", handleScroll);

  // Attach event handler for keys to move to the next/previous image
  window.addEventListener("keydown", (e: any) => {
    if (e.key === "ArrowLeft") {
      moveImage(-1);
    } else if (e.key === "ArrowRight") {
      moveImage(1);
    }
  });
});

onUnmounted(() => {
  // Infinite Scroll
  window.removeEventListener("scroll", handleScroll);

  // Remove event handler for keys to move to the next/previous image
  window.removeEventListener("keydown", (e: any) => {
    if (e.key === "ArrowLeft") {
      moveImage(-1);
    } else if (e.key === "ArrowRight") {
      moveImage(1);
    }
  });
});

function getNamedSampler(value: string) {
  const parsed_string = +value;

  for (const objectKey of Object.keys(diffusersSchedulerTuple)) {
    const val =
      diffusersSchedulerTuple[
        objectKey as keyof typeof diffusersSchedulerTuple
      ];
    if (val === parsed_string) {
      return objectKey;
    }
  }

  return value;
}

refreshImages();
</script>

<style scoped>
.img-slider {
  aspect-ratio: 1/1;
  height: 182px;
  width: auto;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(
    v-bind("settings.data.settings.frontend.image_browser_columns"),
    1fr
  );
  grid-gap: 8px;
}

.top-bar {
  background-color: v-bind("theme?.Card?.color");
}

.image-column {
  display: flex;
  flex-direction: column;
}
</style>
