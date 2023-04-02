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

    <NTabPane name="SD Upscale">
      <SDUpscale />
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
import ControlNet from "@/components/img2img/ControlNet.vue";
import Img2Img from "@/components/img2img/Img2Img.vue";
import Inpainting from "@/components/img2img/Inpainting.vue";
import SDUpscale from "@/components/img2img/SDUpscale.vue";
import WIP from "@/components/WIP.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
