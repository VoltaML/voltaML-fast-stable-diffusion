<template>
    <NCard>
        <NForm>
        <h2>Timers</h2>
        <NFormItem label="Continuous generation timeout (0 for disabled) [ms]">
        <NInputNumber
            v-model:value="settings.defaultSettings.frontend.on_change_timer"
            :min="0"
            :step="50"
        />
        </NFormItem>
        <h2>Saving</h2>
        <NFormItem label="Enable saving to Prompt Sub Folder">
            <NSwitch
                v-model:value="
                    settings.defaultSettings.frontend.save_to_sub_folder
                "
            />
        </NFormItem>
        </NForm>
    </NCard>
</template>

<script setup lang="ts">
import { NForm, NFormItem, NInputNumber, NSwitch } from "naive-ui";
import { watch } from "vue";
import { useSettings } from "../../store/settings";

const settings = useSettings();

watch(settings.defaultSettings.frontend, () => {
    settings.data.settings.frontend.on_change_timer =
        settings.defaultSettings.frontend.on_change_timer;
    settings.data.settings.frontend.save_to_sub_folder =
        settings.defaultSettings.frontend.save_to_sub_folder;
});
</script>
