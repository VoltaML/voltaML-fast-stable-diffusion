<template>
  <NForm>
    <h2>Timers</h2>
    <NFormItem label="Continuous generation timeout (0 for disabled) [ms]">
      <NInputNumber
        v-model:value="settings.defaultSettings.frontend.on_change_timer"
        :min="0"
        :step="50"
      />
    </NFormItem>
  </NForm>
</template>

<script setup lang="ts">
import { NForm, NFormItem, NInputNumber } from "naive-ui";
import { watch } from "vue";
import { useSettings } from "../../store/settings";

const settings = useSettings();

watch(settings.defaultSettings.frontend, () => {
  settings.data.settings.frontend.on_change_timer =
    settings.defaultSettings.frontend.on_change_timer;
});
</script>
