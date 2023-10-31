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
import { AutofillManager, DependencyManager } from "@/components";
import { useState } from "@/store/state";
import { NTabPane, NTabs } from "naive-ui";

const state = useState();
</script>
