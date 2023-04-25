<template>
  <NCard title="Stats" v-if="genData.time_taken || genData.seed">
    <NDescriptions>
      <NDescriptionsItem label="Total Time">
        {{ genData.time_taken }}s
      </NDescriptionsItem>
      <NDescriptionsItem label="Seed"> {{ genData.seed }} </NDescriptionsItem>
    </NDescriptions>
  </NCard>
</template>

<script lang="ts" setup>
import { NCard, NDescriptions, NDescriptionsItem } from "naive-ui";
import { defineProps, type PropType } from "vue";
import type { GenerationData } from "../store/state";

defineProps({
  genData: {
    type: Object as PropType<GenerationData>,
    required: true,
  },
});
</script>
