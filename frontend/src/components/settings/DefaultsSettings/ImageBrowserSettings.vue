<template>
  <NForm>
    <NCard>
      <NFormItem label="Number of columns" label-placement="left">
        <NInputNumber
          v-model:value="
            settings.defaultSettings.frontend.image_browser_columns
          "
        />
      </NFormItem>
    </NCard>
  </NForm>
</template>

<script lang="ts" setup>
import { useSettings } from "@/store/settings";
import { NCard, NForm, NFormItem, NInputNumber } from "naive-ui";

const settings = useSettings();
</script>
