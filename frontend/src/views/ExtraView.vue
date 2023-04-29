<template>
  <NTabs type="segment" v-model:value="state.state.extra.tab">
    <NTabPane name="Upscale">
      <RealESRGAN />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import RealESRGAN from "@/components/extra/RealESRGAN.vue";
import { useState } from "@/store/state";
import { NTabPane, NTabs } from "naive-ui";

const state = useState();
</script>
