<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NTabs>
    <NTabPane name="Highres fix">
      <HiresSettings />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import { NTabPane, NTabs } from "naive-ui";
import HiresSettings from "./ExtraSettings/HiresSettings.vue";
</script>
