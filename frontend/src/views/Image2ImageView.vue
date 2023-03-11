<template>
  <NTabs type="segment">
    <NTabPane name="Image to Image">
      <Img2Img />
    </NTabPane>

    <NTabPane name="ControlNet">
      <ControlNet />
    </NTabPane>

    <NTabPane name="Inpainting">
      <Inpainting />
    </NTabPane>

    <NTabPane name="Image variations">
      <ImageVariations />
    </NTabPane>

    <NTabPane name="SD Upscale">
      <WIP />
    </NTabPane>

    <NTabPane name="Depth to Image">
      <WIP />
    </NTabPane>

    <NTabPane name="Pix to Pix">
      <WIP />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import ControlNet from "@/components/ControlNet.vue";
import ImageVariations from "@/components/ImageVariations.vue";
import Img2Img from "@/components/Img2Img.vue";
import Inpainting from "@/components/Inpainting.vue";
import WIP from "@/components/WIP.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
