<template>
  <NCard title="Output" hoverable>
    <div style="width: 100%; display: flex; justify-content: center">
      <NImage
        v-if="displayedImage"
        :src="displayedImage.toString()"
        :img-props="{
          style: 'max-width: 100%; max-height: 70vh; width: 100%',
        }"
        style="max-width: 100%; max-height: 60vh; width: 100%; height: 100%"
        object-fit="contain"
      />
    </div>
    <div style="height: 150px; margin-top: 12px" v-if="images.length > 1">
      <NScrollbar x-scrollable>
        <span
          v-for="(image, i) in props.images"
          v-bind:key="i"
          @click="$emit('image-clicked', image.toString())"
          style="cursor: pointer"
        >
          <img
            :src="image.toString()"
            style="
              height: 100px;
              width: 100px;
              margin: 5px;
              object-fit: contain;
            "
          />
        </span>
      </NScrollbar>
    </div>
  </NCard>
</template>

<script lang="ts" setup>
import { NCard, NImage, NScrollbar } from "naive-ui";
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

defineEmits(["image-clicked"]);

const displayedImage = computed(() => {
  if (props.currentImage) {
    return props.currentImage;
  } else if (props.images.length > 0) {
    return props.images[0];
  } else {
    return "";
  }
});
</script>
