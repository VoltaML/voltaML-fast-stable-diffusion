<template>
  <NTabs type="segment">
    <NTabPane name="AITemplate">
      <AITemplateAccelerate />
    </NTabPane>
    <NTabPane name="Dynamic AITemplate">
      <AITemplateDynamicAccelerate />
    </NTabPane>
    <NTabPane name="ONNX">
      <ONNXAccelerate />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import AITemplateAccelerate from "@/components/accelerate/AITemplateAccelerate.vue";
import AITemplateDynamicAccelerate from "@/components/accelerate/AITemplateDynamicAccelerate.vue";
import ONNXAccelerate from "@/components/accelerate/ONNXAccelerate.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
