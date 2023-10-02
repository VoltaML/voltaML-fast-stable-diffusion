<template>
  <NTabs type="segment" v-model:value="state.state.img2img.tab">
    <NTabPane tab="Image to Image" name="img2img">
      <Img2Img />
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
import ControlNet from "@/components/img2img/ControlNet.vue";
import Img2Img from "@/components/img2img/Img2Img.vue";
import Inpainting from "@/components/img2img/Inpainting.vue";
import { NTabPane, NTabs } from "naive-ui";
import { useState } from "../store/state";

const state = useState();
</script>
