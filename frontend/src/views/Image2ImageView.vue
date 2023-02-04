<template>
  <NTabs type="segment">
    <NTabPane name="Image to Image">
      <Img2Img />
    </NTabPane>
    <NTabPane name="Image variations">
      <WIP />
    </NTabPane>
    <NTabPane name="Inpaint">
      <WIP />
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
import Img2Img from "@/components/Img2Img.vue";
import WIP from "@/components/WIP.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
