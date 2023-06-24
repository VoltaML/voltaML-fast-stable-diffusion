<template>
  <NCard>
    <NForm>
      <NFormItem label="Theme">
        <NSelect
          :options="themeOptions"
          v-model:value="settings.defaultSettings.frontend.theme"
        />
      </NFormItem>
      <NFormItem label="Enable Theme Editor">
        <NSwitch
          v-model:value="settings.defaultSettings.frontend.enable_theme_editor"
        />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { useSettings } from "@/store/settings";
import {
  NCard,
  NForm,
  NFormItem,
  NSelect,
  NSwitch,
  type SelectOption,
} from "naive-ui";
import { watch } from "vue";

const settings = useSettings();

const themeOptions: SelectOption[] = [
  { label: "Dark", value: "dark" },
  { label: "Light", value: "light" },
];

watch(settings.defaultSettings.frontend, () => {
  settings.data.settings.frontend = settings.defaultSettings.frontend;
});
</script>
