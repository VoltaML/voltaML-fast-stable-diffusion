<template>
  <NCard>
    <NForm>
      <h2>Default settings</h2>
      <NFormItem label="Number of columns">
        <NInputNumber
          v-model:value="
            settings.defaultSettings.frontend.image_browser_columns
          "
        />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { useSettings } from "@/store/settings";
import { NCard, NForm, NFormItem, NInputNumber } from "naive-ui";

const settings = useSettings();
</script>
