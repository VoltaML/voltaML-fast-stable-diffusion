<template>
  <NTabs type="segment">
    <NTabPane name="Huggingface">
      <HuggingfaceDownload />
    </NTabPane>
    <NTabPane name="CivitAI">
      <CivitAIDownload />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import CivitAIDownload from "@/components/download/CivitAIDownload.vue";
import HuggingfaceDownload from "@/components/download/HuggingfaceDownload.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>

<style scoped></style>
