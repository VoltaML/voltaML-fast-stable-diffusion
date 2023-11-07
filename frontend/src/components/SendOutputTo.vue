<template>
  <NModal v-model:show="showModal" mask-closable close-on-esc>
    <NCard style="max-width: 700px" title="Copy additional properties">
      <div
        style="
          display: flex;
          flex-direction: row;
          justify-content: flex-end;
          margin-bottom: 8px;
        "
      >
        <NButton
          type="success"
          ghost
          style="margin-right: 4px"
          @click="selectAll"
          >Select All</NButton
        >
        <NButton type="warning" ghost @click="selectNone">Select None</NButton>
      </div>
      <NScrollbar style="max-height: 70vh; margin-bottom: 8px">
        <div style="margin: 0 24px">
          <div v-for="item in valuesToCopyFiltered" v-bind:key="item">
            <div
              style="
                display: flex;
                flex-direction: row;
                justify-content: space-between;
              "
            >
              {{ capitalizeAndReplace(item) }}
              <NSwitch
                :value="valuesToCopy[item]"
                @update:value="(v) => (valuesToCopy[item] = v)"
              />
            </div>
            <NDivider style="margin: 12px 0" />
          </div>
        </div>
      </NScrollbar>
      <div
        style="display: flex; flex-direction: row; justify-content: flex-end"
      >
        <NButton
          type="default"
          @click="() => (showModal = false)"
          style="margin-right: 4px; flex-grow: 1"
        >
          <template #icon>
            <CloseOutline />
          </template>
          Cancel
        </NButton>
        <NButton type="primary" @click="modalCopyClick" style="flex-grow: 1">
          <template #icon>
            <CopyOutline />
          </template>
          Copy
        </NButton>
      </div>
    </NCard>
  </NModal>

  <div v-if="output">
    <NCard style="margin: 12px 0" title="Send To" v-if="output && card">
      <NGrid cols="4" x-gap="4" y-gap="4">
        <NGi v-for="target in Object.keys(targets)" v-bind:key="target">
          <NButton
            type="default"
            @click="() => handleClick(target as keyof typeof targets)"
            style="width: 100%"
            ghost
            >{{ capitalizeAndReplace(target) }}</NButton
          >
        </NGi>
      </NGrid>
    </NCard>
    <NGrid cols="3" x-gap="4" y-gap="4" v-else>
      <NGi v-for="target in Object.keys(targets)" v-bind:key="target">
        <NButton
          type="default"
          @click="() => handleClick(target as keyof typeof targets)"
          style="width: 100%"
          ghost
          >-> {{ capitalizeAndReplace(target) }}</NButton
        >
      </NGi>
    </NGrid>
  </div>
</template>

<script lang="ts" setup>
import { useState } from "@/store/state";
import { CloseOutline, CopyOutline } from "@vicons/ionicons5";
import {
  NButton,
  NCard,
  NDivider,
  NGi,
  NGrid,
  NModal,
  NScrollbar,
  NSwitch,
} from "naive-ui";
import { computed, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useSettings } from "../store/settings";
const router = useRouter();

const settings = useSettings();
const state = useState();

const showModal = ref(false);
const maybeTarget = ref<keyof typeof targets | null>(null);

// Key is both name and tab name, value is target url
const targets = {
  txt2img: "txt2img",
  img2img: "img2img",
  controlnet: "img2img",
  inpainting: "img2img",
  upscale: "imageProcessing",
  tagger: "tagger",
} as const;

const props = defineProps({
  output: {
    type: String,
    required: true,
  },
  card: {
    type: Boolean,
    default: true,
  },
  data: {
    type: Object,
    required: false,
    default: () => ({}),
  },
});

function handleClick(target: keyof typeof targets) {
  if (props.data) {
    maybeTarget.value = target;
    showModal.value = true;
  } else {
    toTarget(target);
  }
}

function modalCopyClick() {
  showModal.value = false;
  if (maybeTarget.value) {
    const tmp = maybeTarget.value;
    maybeTarget.value = null;
    toTarget(tmp);
  }
}

// Boolean map of settings to copy
const valuesToCopy = reactive(
  Object.fromEntries(Object.keys(props.data).map((key) => [key, false]))
);

watch(
  () => props.data,
  (newData) => {
    Object.keys(newData).forEach((key) => {
      // eslint-disable-next-line no-prototype-builtins
      if (!valuesToCopy.hasOwnProperty(key)) {
        valuesToCopy[key] = false;
      }
    });
  }
);

const valuesToCopyFiltered = computed(() => {
  return Object.keys(valuesToCopy).filter((key) => {
    if (maybeTarget.value) {
      return Object.keys(settings.data.settings[maybeTarget.value]).includes(
        key
      );
    }
  });
});

async function toTarget(target: keyof typeof targets) {
  const targetPage = targets[target];

  if (target !== "txt2img") {
    settings.data.settings[target].image = props.output;
  }

  if (targetPage !== "txt2img" && target !== "txt2img") {
    state.state[targetPage].tab = target;
  }

  Object.keys(props.data).forEach((key) => {
    if (valuesToCopy[key]) {
      if (Object.keys(settings.data.settings[target]).includes(key)) {
        // @ts-ignore
        settings.data.settings[target][key] = props.data[key];
      }
    }
  });

  await router.push("/" + targetPage);
}

function capitalizeAndReplace(target: string) {
  return target
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function selectAll() {
  for (const key in valuesToCopy) {
    valuesToCopy[key] = true;
  }
}

function selectNone() {
  for (const key in valuesToCopy) {
    valuesToCopy[key] = false;
  }
}
</script>
