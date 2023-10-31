<template>
  <NTabs type="segment">
    <NTabPane name="Dynamic AITemplate">
      <AITemplateDynamicAccelerate />
    </NTabPane>
    <NTabPane name="ONNX">
      <ONNXAccelerate />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import { AITemplateDynamicAccelerate, ONNXAccelerate } from "@/components";
import { NTabPane, NTabs } from "naive-ui";
</script>
