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
    <NTabPane name="Theme">
      <ThemeSettings />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import { NTabPane, NTabs } from "naive-ui";
import ControlNetSettings from "./FrontendSettings/ControlNetSettings.vue";
import ImageBrowserSettings from "./FrontendSettings/ImageBrowserSettings.vue";
import ImageToImageSettings from "./FrontendSettings/ImageToImageSettings.vue";
import InpaintingSettings from "./FrontendSettings/InpaintingSettings.vue";
import TextToImageSettings from "./FrontendSettings/TextToImageSettings.vue";
import ThemeSettings from "./FrontendSettings/ThemeSettings.vue";
</script>
