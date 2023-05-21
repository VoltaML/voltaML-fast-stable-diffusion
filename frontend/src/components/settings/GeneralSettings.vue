<template>
  <NForm>
    <h2>Timers</h2>
    <NFormItem label="Continuous generation timeout (0 for disabled)">
      <NInputNumber
        v-model:value="settings.defaultSettings.frontend.on_change_timer"
        :min="100"
        :step="50"
      />
    </NFormItem>
  </NForm>
</template>

<script setup lang="ts">
import { NForm, NFormItem, NInputNumber } from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
