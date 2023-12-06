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
    <NButton
      @click="refreshImages"
      style="margin-right: 24px; padding: 0px 48px"
      type="primary"
    >
      <NIcon>
        <SearchOutline />
      </NIcon>
    </NButton>
  </div>
  <div class="main-container" style="margin: 12px; margin-top: 8px">
    <div ref="scrollComponent">
      <div class="image-grid" ref="gridRef">
        <div
          v-for="(item, item_index) in modelData"
          v-bind:key="item_index"
          style="
            border-radius: 20px;
            position: relative;
            border: 1px solid #505050;
            overflow: hidden;
            margin-bottom: 8px;
          "
        >
          <CivitAIModelImage
            :item="item"
            :item_index="item_index"
            @img-click="imgClick"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { CivitAIModelImage, ModelPopup } from "@/components";
import { themeOverridesKey } from "@/injectionKeys";
import { SearchOutline } from "@vicons/ionicons5";
import { NButton, NIcon, NInput, NSelect, useLoadingBar } from "naive-ui";
import { inject, onMounted, onUnmounted, reactive, ref, type Ref } from "vue";
import type { ICivitAIModel, ICivitAIModels } from "../../civitai";

const theme = inject(themeOverridesKey);

const loadingLock = ref(false);
const currentPage = ref(1);
const sortBy = ref("Most Downloaded");
const types: Ref<"Checkpoint" | "TextualInversion" | "LORA" | ""> = ref("");

const currentModel = ref<ICivitAIModel | null>(null);
const showModal = ref(false);
const gridRef = ref<HTMLElement | null>(null);

const scrollComponent = ref<HTMLElement | null>(null);

const itemFilter = ref("");

const currentIndex = ref(0);

const loadingBar = useLoadingBar();

function imgClick(item_index: number) {
  const item = modelData[item_index];

  currentIndex.value = item_index;
  currentModel.value = item;
  showModal.value = true;
}

const modelData: ICivitAIModel[] = reactive<ICivitAIModel[]>([]);

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
  const bottombbox = gridRef.value!.getBoundingClientRect().bottom;
  if (minBox === 0) {
    minBox = bottombbox;
  } else if (bottombbox < minBox) {
    minBox = bottombbox;
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
  if (currentModel.value === null) {
    return;
  }

  if (direction === -1) {
    // Traverse all the columns before removing one from the currentIndexOfColumn
    if (currentIndex.value > 0) {
      imgClick(currentIndex.value - 1);
    }
  } else if (direction === 1) {
    // Traverse all the columns before adding one from the currentIndexOfColumn
    if (currentIndex.value < modelData.length - 1) {
      imgClick(currentIndex.value + 1);
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
.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  grid-gap: 8px;
}

.top-bar {
  background-color: v-bind("theme?.Card?.color");
}
</style>
