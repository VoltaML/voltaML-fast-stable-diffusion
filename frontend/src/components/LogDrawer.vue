<template>
  <NDrawer
    placement="bottom"
    v-model:show="glob.state.log_drawer.enabled"
    :auto-focus="false"
    :show-mask="true"
    height="70vh"
  >
    <NDrawerContent closable title="Log - 500 latest messages">
      <NLog ref="logRef" :log="log" trim style="height: 100%" />
    </NDrawerContent>
  </NDrawer>
</template>

<script setup lang="ts">
import { useState } from "@/store/state";
import { NDrawer, NDrawerContent, NLog } from "naive-ui";
import { computed } from "vue";

const glob = useState();
const log = computed(() => glob.state.log_drawer.logs.join("\n"));
</script>
