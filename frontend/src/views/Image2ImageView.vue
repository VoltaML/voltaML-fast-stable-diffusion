<template>
  <NTabs type="segment" v-model:value="state.state.img2img.tab">
    <NTabPane tab="Image to Image" name="img2img">
      <ImageToImage />
    </NTabPane>

    <NTabPane tab="ControlNet" name="controlnet">
      <ControlNet />
    </NTabPane>

    <NTabPane tab="Inpainting" name="inpainting">
      <Inpainting />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import { ControlNet, ImageToImage, Inpainting } from "@/components";
import { NTabPane, NTabs } from "naive-ui";
import { useState } from "../store/state";

const state = useState();
</script>
