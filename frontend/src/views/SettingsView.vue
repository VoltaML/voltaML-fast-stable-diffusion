<template>
  <div class="main-container">
    <NCard>
      <NTabs>
        <NTabPane name="Frontend">
          <FrontendSettings />
        </NTabPane>
        <NTabPane name="API">
          <APISettings />
        </NTabPane>
        <NTabPane name="Bot">
          <BotSettings />
        </NTabPane>
        <NTabPane name="General">
          <GeneralSettings />
        </NTabPane>
        <NTabPane name="Extra">
          <ExtraSettings />
        </NTabPane>

        <template #suffix>
          <NButton
            type="error"
            ghost
            style="margin-right: 12px"
            @click="resetSettings"
            >Reset Settings</NButton
          >
          <NButton type="success" ghost @click="saveSettings"
            >Save Settings</NButton
          >
        </template>
      </NTabs>
    </NCard>
  </div>
</template>

<script lang="ts" setup>
import APISettings from "@/components/settings/APISettings.vue";
import BotSettings from "@/components/settings/BotSettings.vue";
import ExtraSettings from "@/components/settings/ExtraSettings.vue";
import FrontendSettings from "@/components/settings/FrontendSettings.vue";
import GeneralSettings from "@/components/settings/GeneralSettings.vue";
import { serverUrl } from "@/env";
import { defaultSettings } from "@/settings";
import { useSettings } from "@/store/settings";
import { NButton, NCard, NTabPane, NTabs, useMessage } from "naive-ui";

const message = useMessage();
const settings = useSettings();

function resetSettings() {
  // Deepcopy and assign
  Object.assign(
    settings.defaultSettings,
    JSON.parse(JSON.stringify(defaultSettings))
  );

  message.warning(
    "Settings were reset to default values, please save them if you want to keep them"
  );
}

function saveSettings() {
  message.success("Settings Saved");
  console.log(settings.defaultSettings);
  fetch(`${serverUrl}/api/settings/save`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(settings.defaultSettings),
  });
}
</script>
