<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NCard title="Restoration" class="generate-extra-card">
    <NTabs animated type="segment">
      <NTabPane tab="ADetailer" name="adetailer">
        <ADetailer :tab="props.tab" :target="props.target" />
      </NTabPane>
    </NTabs>
  </NCard>
</template>

<script setup lang="ts">
import { ADetailer } from "@/components";
import type { InferenceTabs } from "@/types";
import { NCard, NTabPane, NTabs } from "naive-ui";
import type { PropType } from "vue";

const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
  target: {
    type: String as PropType<"settings" | "defaultSettings">,
    required: false,
    default: "settings",
  },
});
</script>
