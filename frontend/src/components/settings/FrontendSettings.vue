<template>
  <NTabs>
    <NTabPane name="Text to Image">
      <TextToImageSettings />
    </NTabPane>
    <NTabPane name="Image to Image">
      <ImageToImageSettings />
    </NTabPane>
    <NTabPane name="ControlNet">
      <ControlNetSettings />
    </NTabPane>
    <NTabPane name="Inpainting">
      <InpaintingSettings />
    </NTabPane>
    <NTabPane name="Image Browser">
      <ImageBrowserSettings />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import {
  ControlNetSettings,
  ImageBrowserSettings,
  ImageToImageSettings,
  InpaintingSettings,
  TextToImageSettings,
} from "@/components";
import { NTabPane, NTabs } from "naive-ui";
</script>
./defaultSettings/ControlNetSettings.vue./defaultSettings/ImageBrowserSettings.vue./defaultSettings/ImageToImageSettings.vue./defaultSettings/InpaintingSettings.vue./defaultSettings/TextToImageSettings.vue
