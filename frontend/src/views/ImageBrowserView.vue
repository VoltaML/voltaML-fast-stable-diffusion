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
      v-model:value="conf.data.settings.frontend.image_browser_columns"
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
      preset="card"
      style="width: 85vw"
      title="Image Info"
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
  NInput,
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
const itemFilter = ref("");

const gridColumnRefs = ref<HTMLElement[]>([]);

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
      global.state.imageBrowser.currentImageMetadata = new Map<
        string,
        string
      >();
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

function imgClick(column_index: number, item_index: number) {
  const item = columns.value[column_index][item_index];
  global.state.imageBrowser.currentImage = item;
  setByte64FromImage(item.path);
  const url = new URL(`${serverUrl}/api/output/data/`);
  url.searchParams.append("filename", item.path);
  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      global.state.imageBrowser.currentImageMetadata = data;
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
  for (let i = 0; i < conf.data.settings.frontend.image_browser_columns; i++) {
    cols.push([]);
  }
  for (let i = 0; i < computedImgDataLimit.value; i++) {
    cols[i % conf.data.settings.frontend.image_browser_columns].push(
      filteredImgData.value[i]
    );
  }
  return cols;
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

.image-grid {
  display: grid;
  grid-template-columns: repeat(
    v-bind("conf.data.settings.frontend.image_browser_columns"),
    1fr
  );
  grid-gap: 8px;
}

.top-bar {
  background-color: v-bind(backgroundColor);
}

.image-column {
  display: flex;
  flex-direction: column;
}
</style>
