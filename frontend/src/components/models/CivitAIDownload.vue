<template>
  <ModelPopup
    :model="currentModel"
    :show-modal="showModal"
    @update:show-modal="showModal = $event"
  />

  <div
    style="
      width: calc(100vw - 98px);
      height: 48px;
      border-bottom: #505050 1px solid;
      display: flex;
      justify-content: end;
      align-items: center;
      padding-right: 24px;
      position: sticky;
      top: 52px;
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
  <div class="main-container" style="margin: 12px; margin-top: 52px">
    <div ref="scrollComponent">
      <div class="image-grid">
        <div
          v-for="(column, column_index) in columns"
          v-bind:key="column_index"
          class="image-column"
          ref="gridColumnRefs"
        >
          <div
            v-for="(item, item_index) in column"
            v-bind:key="item_index"
            style="border-radius: 20px; overflow: hidden"
          >
            <img
              :src="item.modelVersions[0].images[0].url"
              :style="{
                width: '100%',
                height: 'auto',
                borderRadius: '8px',
                cursor: 'pointer',
                marginBottom: '6px',
                filter:
                  item.modelVersions[0].images[0].nsfw !== 'None'
                    ? 'blur(12px)'
                    : 'none',
              }"
              @click="imgClick(column_index, item_index)"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import ModelPopup from "@/components/models/ModelPopup.vue";
import { GridOutline } from "@vicons/ionicons5";
import { NIcon, NInput, NSlider, useLoadingBar } from "naive-ui";
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import type { ICivitAIModel, ICivitAIModels } from "../../civitai";
import { useSettings } from "../../store/settings";

const conf = useSettings();

const loadingLock = ref(false);
const currentPage = ref(1);

const currentModel = ref<ICivitAIModel | null>(null);
const showModal = ref(false);

const scrollComponent = ref<HTMLElement | null>(null);

const itemFilter = ref("");

const gridColumnRefs = ref<HTMLElement[]>([]);

const currentColumn = ref(0);
const currentRowIndex = ref(0);

const loadingBar = useLoadingBar();

function imgClick(column_index: number, item_index: number) {
  currentRowIndex.value = item_index;
  currentColumn.value = column_index;
  const item = columns.value[column_index][item_index];

  currentModel.value = item;
  showModal.value = true;
}

const modelData: ICivitAIModel[] = reactive<ICivitAIModel[]>([]);

const columns = computed(() => {
  const cols: ICivitAIModel[][] = [];
  for (let i = 0; i < conf.data.settings.frontend.image_browser_columns; i++) {
    cols.push([]);
  }
  for (let i = 0; i < modelData.length; i++) {
    cols[i % conf.data.settings.frontend.image_browser_columns].push(
      modelData[i]
    );
  }
  return cols;
});

async function refreshImages() {
  // Clear the data
  modelData.splice(0, modelData.length);

  // Fetch new data
  const url = new URL("https://civitai.com/api/v1/models");
  url.searchParams.append("sort", "Most Downloaded");
  await fetch(url)
    .then((res) => res.json())
    .then((data: ICivitAIModels) => {
      data.items.forEach((item) => {
        modelData.push(item);
      });
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
    if (loadingLock.value) {
      return;
    }

    loadingLock.value = true;
    currentPage.value++;
    loadingBar.start();

    const pageToFetch = currentPage.value.toString();
    const url = new URL("https://civitai.com/api/v1/models");
    url.searchParams.append("sort", "Most Downloaded");
    url.searchParams.append("page", pageToFetch);
    console.log("Fetching page: " + url.toString());
    fetch(url)
      .then((res) => res.json())
      .then((data: ICivitAIModels) => {
        data.items.forEach((item) => {
          modelData.push(item);
        });
        loadingBar.finish();
        loadingLock.value = false;
      })
      .catch((err) => {
        console.error(err);
        loadingBar.error();
        loadingLock.value = false;
      });
  }
};

function moveImage(direction: number) {
  const numColumns = conf.data.settings.frontend.image_browser_columns;

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
