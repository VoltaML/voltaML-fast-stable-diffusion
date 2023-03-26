<template>
  <NTabs type="segment">
    <NTabPane name="Real-ESRGAN">
      <RealESRGAN />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import RealESRGAN from "@/components/extra/RealESRGAN.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
