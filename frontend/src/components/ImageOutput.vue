<template>
  <NCard title="Output" hoverable>
    <div style="width: 100%; display: flex; justify-content: center">
      <NImage
        v-if="displayedImage"
        :src="`data:image/png;base64,${displayedImage}`"
        :img-props="{
          style: 'max-width: 100%; max-height: 70vh; width: 100%',
        }"
        style="max-width: 100%; max-height: 60vh; width: 100%; height: 100%"
        object-fit="contain"
      />
    </div>
    <div style="height: 100px; margin-top: 12px" v-if="images.length > 1">
      <NImageGroup>
        <NImage
          v-for="(image, i) in allImages"
          v-bind:key="i"
          :src="`data:image/png;base64,${image}`"
          class="bottom-images"
          :img-props="{
            style: 'height: 100px; width: 100px; margin: 5px;',
          }"
          object-fit="contain"
          @click="() => (displayedImage = image.toString())"
        />
      </NImageGroup>
    </div>
  </NCard>
</template>

<script lang="ts" setup>
import { NCard, NImage, NImageGroup } from "naive-ui";
import { computed, defineProps } from "vue";

const props = defineProps({
  currentImage: {
    type: String,
    required: true,
  },
  images: {
    type: Array<String>,
    required: false,
    default: () => [],
  },
});

const allImages = computed(() => [props.currentImage, ...props.images]);
const displayedImage = computed(() => props.currentImage);
</script>

<style scoped>
/* .bottom-images {
  height: 100px;
  width: 100px;
  margin: 5px;
  object-fit: contain;
} */
</style>
