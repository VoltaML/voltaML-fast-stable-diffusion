<template>
  <NCard title="High Resolution Fix" class="generate-extra-card">
    <NTabs animated type="segment">
      <NTabPane tab="Image to Image" name="highresfix">
        <HighResFix :tab="props.tab" :target="props.target" />
      </NTabPane>
      <NTabPane tab="Scalecrafter" name="scalecrafter">
        <Scalecrafter :tab="props.tab" :target="props.target" />
      </NTabPane>
      <NTabPane tab="DeepShrink" name="deepshrink">
        <DeepShrink :tab="props.tab" :target="props.target" />
      </NTabPane>
    </NTabs>
  </NCard>
</template>

<script setup lang="ts">
import { DeepShrink, HighResFix, Scalecrafter } from "@/components";
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
