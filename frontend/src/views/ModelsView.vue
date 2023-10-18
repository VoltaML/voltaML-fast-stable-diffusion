<template>
  <NTabs type="segment">
    <NTabPane name="Manager">
      <ModelManager />
    </NTabPane>
    <NTabPane name="Huggingface">
      <HuggingfaceDownload />
    </NTabPane>
    <NTabPane name="CivitAI" style="padding-top: 0">
      <CivitAIDownload />
    </NTabPane>
    <NTabPane name="Convert">
      <ModelConvert />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import CivitAIDownload from "@/components/models/CivitAIDownload.vue";
import HuggingfaceDownload from "@/components/models/HuggingfaceDownload.vue";
import ModelConvert from "@/components/models/ModelConvert.vue";
import ModelManager from "@/components/models/ModelManager.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
