<template>
  <NTabs type="segment" v-model:value="state.state.extra.tab">
    <NTabPane tab="Upscale" name="upscale">
      <Upscale />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import Upscale from "@/components/extra/Upscale.vue";
import { useState } from "@/store/state";
import { NTabPane, NTabs } from "naive-ui";

const state = useState();
</script>
