<template>
  <NResult
    title="Work in progress"
    description="This page is still under development."
    style="
      height: 70vh;
      display: flex;
      align-items: center;
      justify-content: center;
    "
  >
    <template #icon>
      <NIcon size="250">
        <BuildOutline />
      </NIcon>
    </template>
  </NResult>
</template>

<script lang="ts" setup>
import { BuildOutline } from "@vicons/ionicons5";
import { NIcon, NResult } from "naive-ui";
</script>
