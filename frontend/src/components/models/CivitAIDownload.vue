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
      placeholder="Filter"
      style="margin-left: 12px; margin-right: 4px"
    />
    <NSelect
      v-model:value="sortBy"
      :options="[
        {
          value: 'Most Downloaded',
          label: 'Most Downloaded',
        },
        {
          value: 'Highest Rated',
          label: 'Highest Rated',
        },
        {
          value: 'Newest',
          label: 'Newest',
        },
      ]"
      style="margin-right: 4px"
    />
    <NSelect
      v-model:value="types"
      :options="[
        {
          value: '',
          label: 'All',
        },
        {
          value: 'Checkpoint',
          label: 'Checkpoint',
        },
        {
          value: 'TextualInversion',
          label: 'Textual Inversion',
        },
        {
          value: 'LORA',
          label: 'LORA',
        },
      ]"
      style="margin-right: 4px"
    />
    <NButton @click="refreshImages" style="margin-right: 24px" type="primary">
      <NIcon>
        <SearchOutline />
      </NIcon>
    </NButton>

    <NIcon style="margin-right: 12px" size="22">
      <GridOutline />
    </NIcon>
    <NSlider
      style="width: 30vw"
      :min="1"
      :max="10"
      v-model:value="settings.data.settings.frontend.image_browser_columns"
    >
    </NSlider>
  </div>
  <div class="main-container" style="margin: 12px; margin-top: 8px">
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
            style="
              border-radius: 20px;
              position: relative;
              border: 1px solid #505050;
              overflow: hidden;
              margin-bottom: 8px;
            "
          >
            <div v-if="item.modelVersions[0].images[0]?.url">
              <img
                :src="item.modelVersions[0].images[0].url"
                :style="{
                  width: '100%',
                  height: 'auto',
                  minHeight: '200px',
                  cursor: 'pointer',
                  borderRadius: '8px',
                  filter:
                    nsfwIndex(item.modelVersions[0].images[0].nsfw) >
                    settings.data.settings.frontend.nsfw_ok_threshold
                      ? 'blur(12px)'
                      : 'none',
                }"
                @click="imgClick(column_index, item_index)"
              />
              <div
                style="
                  position: absolute;
                  width: 100%;
                  bottom: 0;
                  padding: 0 8px;
                  min-height: 32px;
                  overflow: hidden;
                  box-sizing: border-box;
                  backdrop-filter: blur(12px);
                "
              >
                <NText :depth="2">
                  {{ item.name }}
                </NText>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import ModelPopup from "@/components/models/ModelPopup.vue";
import { themeOverridesKey } from "@/injectionKeys";
import { GridOutline, SearchOutline } from "@vicons/ionicons5";
import {
  NButton,
  NIcon,
  NInput,
  NSelect,
  NSlider,
  NText,
  useLoadingBar,
} from "naive-ui";
import {
  computed,
  inject,
  onMounted,
  onUnmounted,
  reactive,
  ref,
  type Ref,
} from "vue";
import type { ICivitAIModel, ICivitAIModels } from "../../civitai";
import { nsfwIndex } from "../../civitai";
import { useSettings } from "../../store/settings";

const settings = useSettings();
const theme = inject(themeOverridesKey);

const loadingLock = ref(false);
const currentPage = ref(1);
const sortBy = ref("Most Downloaded");
const types: Ref<"Checkpoint" | "TextualInversion" | "LORA" | ""> = ref("");

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
  for (
    let i = 0;
    i < settings.data.settings.frontend.image_browser_columns;
    i++
  ) {
    cols.push([]);
  }
  for (let i = 0; i < modelData.length; i++) {
    cols[i % settings.data.settings.frontend.image_browser_columns].push(
      modelData[i]
    );
  }
  return cols;
});

async function refreshImages() {
  // Clear the data
  currentPage.value = 1;
  modelData.splice(0, modelData.length);

  loadingBar.start();

  // Fetch new data
  const url = new URL("https://civitai.com/api/v1/models");
  url.searchParams.append("sort", sortBy.value);
  if (itemFilter.value !== "") {
    url.searchParams.append("query", itemFilter.value);
  }
  if (types.value) {
    url.searchParams.append("types", types.value);
  }

  await fetch(url)
    .then((res) => res.json())
    .then((data: ICivitAIModels) => {
      data.items.forEach((item) => {
        modelData.push(item);
      });
      loadingBar.finish();
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
    url.searchParams.append("sort", sortBy.value);
    url.searchParams.append("page", pageToFetch);
    if (itemFilter.value !== "") {
      url.searchParams.append("query", itemFilter.value);
    }
    if (types.value) {
      url.searchParams.append("types", types.value);
    }

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

  window.addEventListener("keyup", async (e: KeyboardEvent) => {
    if (e.key === "Enter") {
      await refreshImages();
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

  window.removeEventListener("keyup", async (e: KeyboardEvent) => {
    if (e.key === "Enter") {
      await refreshImages();
    }
  });
});

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
