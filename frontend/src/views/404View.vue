<template>
  <div
    style="
      width: 100vw;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(4px);
    "
  >
    <NCard style="max-width: 40vw; border-radius: 12px">
      <NResult
        status="404"
        title="You got lucky, this page doesn't exist!"
        description="Next time, there will be a rickroll."
        size="large"
      />
    </NCard>
  </div>
</template>

<script lang="ts" setup>
import { NCard, NResult } from "naive-ui";
</script>
