<template>
  <NTabs type="segment" v-model:value="state.state.extra.tab">
    <NTabPane tab="Dependencies" name="dependencies">
      <DependencyManager />
    </NTabPane>
    <NTabPane tab="Autofill" name="autofill">
      <AutofillManager />
    </NTabPane>
  </NTabs>
</template>

<script lang="ts" setup>
import AutofillManager from "@/components/extra/AutofillManager.vue";
import DependencyManager from "@/components/extra/DependencyManager.vue";
import { useState } from "@/store/state";
import { NTabPane, NTabs } from "naive-ui";

const state = useState();
</script>
