<template>
  <NTabs type="segment" v-model:value="state.state.modelManager.tab">
    <NTabPane name="manager" tab="Manager">
      <ModelManager />
    </NTabPane>
    <NTabPane name="huggingface" tab="Huggingface">
      <HuggingfaceDownload />
    </NTabPane>
    <NTabPane name="civitai" tab="CivitAI" style="padding-top: 0">
      <CivitAIDownload />
    </NTabPane>
    <NTabPane name="convert" tab="Convert">
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
import { useState } from "../store/state";

const state = useState();
</script>
