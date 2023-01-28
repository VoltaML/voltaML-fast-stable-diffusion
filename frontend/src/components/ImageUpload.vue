<template>
  <img :src="preview" style="width: 400px; height: auto" />

  <label for="file-upload"
    >Hello world<NButton type="warning" ghost bordered
      >Upload an Image</NButton
    ></label
  >
  <input
    type="file"
    accept="image/*"
    @change="previewImage"
    id="file-upload"
    class="hidden-input"
  />
</template>

<script lang="ts" setup>
import { NButton } from "naive-ui";
import { ref } from "vue";

let preview = ref("");

function previewImage(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const i = e.target?.result;
      if (i) {
        preview.value = i.toString();
      }
    };
    reader.readAsDataURL(input.files[0]);
  }
}
</script>

<style scoped>
.hidden-input {
  display: none;
}
</style>
