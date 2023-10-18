<template>
  <NCard>
    <NForm>
      <NFormItem label="Default Scheduler" label-placement="left">
        <NSelect
          :options="settings.scheduler_options"
          v-model:value="settings.defaultSettings.bot.default_scheduler"
        />
      </NFormItem>
      <NFormItem label="Use Default Negative Prompt" label-placement="left">
        <NSwitch
          v-model:value="
            settings.defaultSettings.bot.use_default_negative_prompt
          "
        />
      </NFormItem>
      <NFormItem label="Verbose" label-placement="left">
        <NSwitch v-model:value="settings.defaultSettings.bot.verbose" />
      </NFormItem>
    </NForm>
  </NCard>
</template>

<script lang="ts" setup>
import { NCard, NForm, NFormItem, NSelect, NSwitch } from "naive-ui";
import { useSettings } from "../../store/settings";

const settings = useSettings();
</script>
