<template>
  <NGrid cols="2" x-gap="4" style="margin-top: 12px">
    <NGi v-if="props.base64image">
      <NButton type="success" @click="downloadImage" style="width: 100%" ghost
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
        :disabled="props.imagePath === undefined"
      >
        <template #icon>
          <NIcon>
            <TrashBin />
          </NIcon>
        </template>
        Delete</NButton
      >
    </NGi>
  </NGrid>
</template>

<script lang="ts" setup>
import { useState } from "@/store/state";
import { Download, TrashBin } from "@vicons/ionicons5";
import { NButton, NGi, NGrid, NIcon } from "naive-ui";
import { ref } from "vue";

const showDeleteModal = ref(false);

const global = useState();

const props = defineProps({
  imagePath: {
    type: String,
    required: false,
  },
  base64image: {
    type: String,
    required: true,
  },
});

function downloadImage() {
  // Download the image
  const a = document.createElement("a");
  a.href = props.base64image;
  a.download = global.state.imageBrowser.currentImage.id;
  a.target = "_blank";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
</script>
