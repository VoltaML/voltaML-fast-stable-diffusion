<template>
  <NCard>
    <NForm>
      <h2>NSFW</h2>
      <NFormItem
        label="NSFW OK threshold (if you don't get the reference, select `I'm too young to die`)"
      >
        <NSelect
          v-model:value="settings.defaultSettings.frontend.nsfw_ok_threshold"
          :options="[
            {
              label: 'I\'m too young to die',
              value: 0,
            },
            {
              label: 'Hurt me plenty',
              value: 1,
            },
            {
              label: 'Ultra violence',
              value: 2,
            },
            {
              label: 'Nightmare',
              value: 3,
            },
          ]"
        />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { useSettings } from "@/store/settings";
import { NCard, NForm, NFormItem, NSelect } from "naive-ui";
import { watch } from "vue";

const settings = useSettings();

watch(
  () => settings.defaultSettings.frontend.nsfw_ok_threshold,
  (value) => {
    settings.data.settings.frontend.nsfw_ok_threshold = value;
  }
);
</script>
