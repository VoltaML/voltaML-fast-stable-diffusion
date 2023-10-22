<template>
  <NCard>
    <NForm>
      <NFormItem label="Theme" label-placement="left">
        <NSelect
          :options="themeOptions"
          v-model:value="settings.defaultSettings.frontend.theme"
          :loading="themesLoading"
          filterable
        />
      </NFormItem>
      <NFormItem label="Background Image Override" label-placement="left">
        <NInput
          v-model:value="
            settings.defaultSettings.frontend.background_image_override
          "
        />
      </NFormItem>
      <NFormItem label="Enable Theme Editor" label-placement="left">
        <NSwitch
          v-model:value="settings.defaultSettings.frontend.enable_theme_editor"
        />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { convertToTextString } from "@/functions";
import { useSettings } from "@/store/settings";
import {
  NCard,
  NForm,
  NFormItem,
  NInput,
  NSelect,
  NSwitch,
  type SelectOption,
} from "naive-ui";
import { computed, reactive, ref, watch } from "vue";

const settings = useSettings();
const extraThemes = reactive<string[]>([]);

const themeOptions = computed<SelectOption[]>(() => {
  return extraThemes.map((theme) => {
    return { label: convertToTextString(theme), value: theme };
  });
});

const themesLoading = ref(true);
fetch(`${serverUrl}/api/general/themes`)
  .then(async (res) => {
    const data = await res.json();
    extraThemes.push(...data);
    themesLoading.value = false;
  })
  .catch((err) => {
    console.error(err);
    themesLoading.value = false;
  });

watch(settings.defaultSettings.frontend, () => {
  settings.data.settings.frontend = settings.defaultSettings.frontend;
});
</script>
