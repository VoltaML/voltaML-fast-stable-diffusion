<template>
  <NTabs type="segment">
    <NTabPane name="Convert">
      <ModelConvert />
    </NTabPane>
    <NTabPane name="Merge">
      <WIP />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import ModelConvert from "@/components/ModelConvert.vue";
import WIP from "@/components/WIP.vue";
import { NTabPane, NTabs } from "naive-ui";
</script>
