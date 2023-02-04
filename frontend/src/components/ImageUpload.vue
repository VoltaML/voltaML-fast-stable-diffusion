<template>
  <NCard title="Input image">
    <div class="image-container">
      <img :src="$props.preview" style="width: 400px; height: auto" />
    </div>

    <NSpace inline justify="space-between" align="center" style="width: 100%">
      <p>{{ image.width }}x{{ image.height }}</p>
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
import { NCard, NSpace } from "naive-ui";
import { computed, type PropType } from "vue";

const props = defineProps({
  callback: {
    type: Object as unknown as PropType<(base64Image: string) => void>,
  },
  preview: {
    type: String,
  },
});

function previewImage(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const i = e.target?.result;
      if (i) {
        const s = i.toString();
        if (props.callback) {
          props.callback(s);
        }
      }
    };
    reader.readAsDataURL(input.files[0]);
  }
}

const image = computed(() => {
  const img = new Image();
  if (!props.preview) {
    return img;
  }
  img.src = props.preview;
  return img;
});
</script>

<style scoped>
.hidden-input {
  display: none;
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

.image-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  overflow: hidden;
}

.image-container {
  height: 70vh;
  width: 100%;
  display: flex;
  justify-content: center;
}
</style>
