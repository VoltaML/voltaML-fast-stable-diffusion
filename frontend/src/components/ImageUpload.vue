<template>
  <NCard title="Input image">
    <div class="image-container">
      <label
        for="file-upload"
        style="width: 100%; height: 100%; cursor: pointer"
      >
        <span style="width: 100%; height: 100%" @drop.prevent="onDrop">
          <img
            :src="$props.preview"
            style="width: 100%"
            v-if="$props.preview"
          />
          <div
            style="
              margin-bottom: 12px;
              display: flex;
              align-items: center;
              justify-content: center;
              height: 100%;
              widows: 100%;
              border: 1px dashed #666;
            "
            v-else
          >
            <NIcon size="48" :depth="3">
              <CloudUpload />
            </NIcon>
            <p style="margin-left: 12px">Drag and drop or click to upload</p>
          </div>
        </span>
      </label>
    </div>

    <div
      style="
        width: 100%;
        display: inline-flex;
        align-items: center;
        justify-content: space-between;
      "
    >
      <p>{{ width }}x{{ height }}</p>
    </div>
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
import { CloudUpload } from "@vicons/ionicons5";
import { NCard, NIcon } from "naive-ui";
import { computed, onMounted, ref, type PropType } from "vue";

const props = defineProps({
  callback: {
    type: Function as PropType<(base64Image: string) => void>,
  },
  preview: {
    type: String,
  },
});

const image = ref<HTMLImageElement>();
const width = computed(() => (image.value ? image.value?.width : 0));
const height = computed(() => (image.value ? image.value?.height : 0));

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
        const img = new Image();
        img.src = s;
        img.onload = () => {
          image.value = img;
        };
      }
    };
    reader.readAsDataURL(input.files[0]);
  }
}

const emit = defineEmits(["file-dropped"]);

function onDrop(e: DragEvent) {
  console.log(e.dataTransfer?.files);

  // Emit file as string
  if (e.dataTransfer?.files) {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const i = e.target?.result;
      if (i) {
        const s = i.toString();
        if (props.callback) {
          props.callback(s);
        }
        const img = new Image();
        img.src = s;
        img.onload = () => {
          emit("file-dropped", s);
        };
      }
    };
    reader.readAsDataURL(e.dataTransfer.files[0]);
  }
}

function preventDefaults(e: Event) {
  e.preventDefault();
}

const events = ["dragenter", "dragover", "dragleave", "drop"];

onMounted(() => {
  events.forEach((eventName) => {
    document.body.addEventListener(eventName, preventDefaults);
  });
});
</script>

<style scoped>
.hidden-input {
  display: none;
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
