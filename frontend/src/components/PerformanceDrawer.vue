<template>
  <NDrawer
    placement="bottom"
    v-model:show="glob.state.perf_drawer.enabled"
    :auto-focus="false"
    :show-mask="true"
    height="70vh"
  >
    <NDrawerContent closable title="Performance statistics">
      <NCard
        v-for="gpu in global.state.perf_drawer.gpus"
        v-bind:key="gpu.uuid"
        style="margin-bottom: 12px"
      >
        <NSpace inline justify="space-between" style="width: 100%">
          <h3>[{{ gpu.index }}] {{ gpu.name }}</h3>
          <h4>
            {{ gpu.power_draw }} / {{ gpu.power_limit }}W ─
            {{ gpu.temperature }}°C
          </h4>
        </NSpace>
        <div style="width: 100%; display: inline-flex; align-items: center">
          <p style="width: 108px">Utilization</p>
          <NProgress
            :percentage="gpu.utilization"
            type="line"
            indicator-placement="inside"
            style="flex-grow: 1; width: 400px"
          />
        </div>
        <div style="width: 100%; display: inline-flex; align-items: center">
          <p style="width: 108px">Memory</p>
          <NProgress
            :percentage="gpu.memory_usage"
            type="line"
            style="flex-grow: 1; width: 400px"
            color="#63e2b7"
            indicator-placement="inside"
          />
          <p style="align-self: flex-end; margin-left: 12px">
            {{ gpu.memory_used }} / {{ gpu.memory_total }} MB
          </p>
        </div>
      </NCard>
    </NDrawerContent>
  </NDrawer>
</template>

<script setup lang="ts">
import { useState } from "@/store/state";
import { NCard, NDrawer, NDrawerContent, NProgress, NSpace } from "naive-ui";

const global = useState();
const glob = useState();
</script>
