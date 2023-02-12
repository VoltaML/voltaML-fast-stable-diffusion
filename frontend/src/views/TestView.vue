<template>
  <NCard title="Input image">
    <div class="image-container" ref="imageContainer">
      <VueDrawingCanvas
        v-model:image="image"
        :width="width"
        :height="height"
        :backgroundImage="preview"
        :lineWidth="strokeWidth"
        strokeType="dash"
        lineCap="round"
        lineJoin="round"
        :fillShape="false"
        :eraser="eraser"
        color="black"
        ref="canvas"
        saveAs="png"
      />
    </div>

    <NSpace
      inline
      justify="space-between"
      align="center"
      style="width: 100%; margin-top: 12px"
    >
      <div style="display: inline-flex; align-items: center">
        <NButton class="utility-button" @click="undo">
          <NIcon>
            <ArrowUndoSharp />
          </NIcon>
        </NButton>
        <NButton class="utility-button" @click="redo">
          <NIcon>
            <ArrowRedoSharp />
          </NIcon>
        </NButton>
        <NButton class="utility-button" @click="toggleEraser">
          <NIcon v-if="eraser">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              fill="currentColor"
              class="bi bi-eraser"
              viewBox="0 0 16 16"
            >
              <path
                d="M8.086 2.207a2 2 0 0 1 2.828 0l3.879 3.879a2 2 0 0 1 0 2.828l-5.5 5.5A2 2 0 0 1 7.879 15H5.12a2 2 0 0 1-1.414-.586l-2.5-2.5a2 2 0 0 1 0-2.828l6.879-6.879zm2.121.707a1 1 0 0 0-1.414 0L4.16 7.547l5.293 5.293 4.633-4.633a1 1 0 0 0 0-1.414l-3.879-3.879zM8.746 13.547 3.453 8.254 1.914 9.793a1 1 0 0 0 0 1.414l2.5 2.5a1 1 0 0 0 .707.293H7.88a1 1 0 0 0 .707-.293l.16-.16z"
              />
            </svg>
          </NIcon>
          <NIcon v-else>
            <BrushSharp />
          </NIcon>
        </NButton>
        <NButton class="utility-button" @click="clearCanvas">
          <NIcon>
            <TrashBinSharp />
          </NIcon>
        </NButton>
        <NSlider
          v-model:value="strokeWidth"
          :min="1"
          :max="50"
          :step="1"
          style="width: 100px; margin: 0 8px"
        />
      </div>
      <label for="file-upload">
        <span class="file-upload">Select image</span>
      </label>
    </NSpace>
    <input
      type="file"
      accept="image/*"
      @change="previewImage"
      id="file-upload"
      class="hidden-input"
    />
  </NCard>
</template>

<script lang="ts" setup>
import {
  ArrowRedoSharp,
  ArrowUndoSharp,
  BrushSharp,
  TrashBinSharp,
} from "@vicons/ionicons5";
import { NButton, NCard, NIcon, NSlider, NSpace } from "naive-ui";
import { ref } from "vue";
import VueDrawingCanvas from "vue-drawing-canvas";

const canvas = ref<InstanceType<typeof VueDrawingCanvas>>();

const width = ref(512);
const height = ref(512);

const strokeWidth = ref(10);
const eraser = ref(false);
const preview = ref("");

const imageContainer = ref<HTMLElement>();

function previewImage(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const i = e.target?.result;
      if (i) {
        const s = i.toString();
        preview.value = s;
        const img = new Image();
        img.src = s;
        img.onload = () => {
          const containerWidth = imageContainer.value?.clientWidth;

          // Scale to fit container
          const containerScaledWidth = containerWidth || img.width;
          const containerScaledHeight =
            (img.height * containerScaledWidth) / containerScaledWidth;

          // Scale to fit into 70vh of the screen
          const screenHeight = window.innerHeight;
          const screenHeightScaledHeight =
            (containerScaledHeight * 0.7 * screenHeight) /
            containerScaledHeight;

          // Scale width to keep aspect ratio
          const screenHeightScaledWidth =
            (img.width * screenHeightScaledHeight) / img.height;

          // Select smaller of the two to fit into the container
          if (containerScaledWidth < screenHeightScaledWidth) {
            width.value = containerScaledWidth;
            height.value = containerScaledHeight;
          } else {
            width.value = screenHeightScaledWidth;
            height.value = screenHeightScaledHeight;
          }

          canvas.value?.redraw(false);
        };
      }
    };
    reader.readAsDataURL(input.files[0]);
  }
}

async function clearCanvas() {
  canvas.value?.reset();
}

function undo() {
  canvas.value?.undo();
}

function redo() {
  canvas.value?.redo();
}

function toggleEraser() {
  console.log(eraser.value);
  eraser.value = !eraser.value;
  console.log(eraser.value);
}

const image = ref("");
</script>

<style scoped>
.hidden-input {
  display: none;
}

.utility-button {
  margin-right: 8px;
}

.file-upload {
  appearance: none;
  background-color: transparent;
  border: 1px solid #63e2b7;
  border-radius: 6px;
  box-shadow: rgba(27, 31, 35, 0.1) 0 1px 0;
  box-sizing: border-box;
  color: #63e2b7;
  cursor: pointer;
  display: inline-block;
  padding: 6px 16px;
  position: relative;
  text-align: center;
  text-decoration: none;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: middle;
  white-space: nowrap;
}

.file-upload:focus:not(:focus-visible):not(.focus-visible) {
  box-shadow: none;
  outline: none;
}

.file-upload:focus {
  box-shadow: rgba(46, 164, 79, 0.4) 0 0 0 3px;
  outline: none;
}

.file-upload:disabled {
  background-color: #94d3a2;
  border-color: rgba(27, 31, 35, 0.1);
  color: rgba(255, 255, 255, 0.8);
  cursor: default;
}

.image-container {
  width: 100%;
  display: flex;
  justify-content: center;
}
</style>
