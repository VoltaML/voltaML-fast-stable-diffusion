<template>
  <NModal :show="global.state.settings_diff.active">
    <NCard title="Settings Diff Resolver" style="max-width: 90vw">
      <NAlert show-icon type="warning">Failed to save config</NAlert>

      <div style="margin: 16px 0">
        <b>Key in question:</b> {{ global.state.settings_diff.key.join("->") }}
        <br />
        <b>Current value</b> {{ global.state.settings_diff.current_value }}
        <br />
        <b>Default value</b> {{ global.state.settings_diff.default_value }}
        <br />
      </div>

      <NGrid cols="2" x-gap="8">
        <NGi>
          <NButton
            type="warning"
            block
            ghost
            style="width: 100%"
            @click="global.state.settings_diff.active = false"
          >
            I Will Fix It Myself
          </NButton>
        </NGi>
        <NGi>
          <NButton type="primary" style="width: 100%" @click="apply">
            Apply Default Value and Save
          </NButton>
        </NGi>
      </NGrid>
    </NCard>
  </NModal>
</template>

<script lang="ts" setup>
import {
  NAlert,
  NButton,
  NCard,
  NGi,
  NGrid,
  NModal,
  useMessage,
} from "naive-ui";
import { useSettings } from "../../store/settings";
import { useState } from "../../store/state";

const global = useState();
const settings = useSettings();
const message = useMessage();

function apply() {
  const key = global.state.settings_diff.key;
  const default_value = global.state.settings_diff.default_value;
  const indexable_keys = key.slice(0, key.length - 1);
  const last_key = key[key.length - 1];

  let current = settings.defaultSettings;
  for (const indexable_key of indexable_keys) {
    // @ts-ignore
    current = current[indexable_key];
  }
  // @ts-ignore
  current[last_key] = default_value;

  settings
    .saveSettings()
    .then(() => {
      message.success("Settings saved");
    })
    .catch((e) => {
      message.error("Failed to save settings: " + e);
    })
    .finally(() => {
      global.state.settings_diff.active = false;
    });
}
</script>
