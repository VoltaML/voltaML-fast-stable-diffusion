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
import { NTabPane, NTabs } from "naive-ui";
import ControlNetSettings from "./DefaultsSettings/ControlNetSettings.vue";
import ImageBrowserSettings from "./DefaultsSettings/ImageBrowserSettings.vue";
import ImageToImageSettings from "./DefaultsSettings/ImageToImageSettings.vue";
import InpaintingSettings from "./DefaultsSettings/InpaintingSettings.vue";
import TextToImageSettings from "./DefaultsSettings/TextToImageSettings.vue";
</script>
