<template>
  <NTabs type="segment">
    <NTabPane name="AITemplate">
      <AITemplateAccelerate />
    </NTabPane>
    <NTabPane name="ONNX">
      <ONNXAccelerate />
    </NTabPane>
    <NTabPane name="TensorRT">
      <TensorRTAccelerate />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import AITemplateAccelerate from "@/components/AITemplateAccelerate.vue";
import ONNXAccelerate from "@/components/ONNXAccelerate.vue";
import TensorRTAccelerate from "@/components/TensorRTAccelerate.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
