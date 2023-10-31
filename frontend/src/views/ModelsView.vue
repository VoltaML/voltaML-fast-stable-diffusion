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
import {
  CivitAIDownload,
  HuggingfaceDownload,
  ModelConvert,
  ModelManager,
} from "@/components";
import { NTabPane, NTabs } from "naive-ui";
import { useState } from "../store/state";

const state = useState();
</script>
