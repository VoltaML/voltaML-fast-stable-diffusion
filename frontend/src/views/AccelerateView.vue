<template>
  <NTabs type="segment">
    <NTabPane name="AITemplate">
      <AITemplateAccelerate />
    </NTabPane>
    <NTabPane name="ONNX">
      <ONNXAccelerate />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import AITemplateAccelerate from "@/components/AITemplateAccelerate.vue";
import ONNXAccelerate from "@/components/ONNXAccelerate.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
