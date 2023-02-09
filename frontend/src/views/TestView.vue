<template>
  <NCard title="Input image">
    <div class="image-container">
      <div :style="{ width: width, height: height }">
        <VueDrawingCanvas
          :width="width"
          :height="height"
          :backgroundImage="preview"
          :lineWidth="strokeWidth"
          color="black"
          ref="canvas"
        />
      </div>
    </div>

    <NSpace inline justify="space-between" align="center" style="width: 100%">
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
        <p>{{ width }}x{{ height }}</p>
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
  TrashBinSharp,
} from "@vicons/ionicons5";
import { NButton, NCard, NIcon, NSlider, NSpace } from "naive-ui";
import { ref } from "vue";
import VueDrawingCanvas from "vue-drawing-canvas";

const canvas = ref<InstanceType<typeof VueDrawingCanvas>>();

const width = ref(512);
const height = ref(512);

const strokeWidth = ref(10);

const preview = ref("");

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
          width.value = img.width;
          height.value = img.height;
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
