<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <NInput
    v-model:value="settings.data.settings[props.tab].prompt"
    type="textarea"
    placeholder="Prompt"
    show-count
    @keyup="
      promptHandleKeyUp(
        $event,
        settings.data.settings[props.tab],
        'prompt',
        state
      )
    "
    @keydown="promptHandleKeyDown"
  />
  <NInput
    v-model:value="settings.data.settings[props.tab].negative_prompt"
    type="textarea"
    placeholder="Negative prompt"
    show-count
    @keyup="
      promptHandleKeyUp(
        $event,
        settings.data.settings[props.tab],
        'negative_prompt',
        state
      )
    "
    @keydown="promptHandleKeyDown"
  />
</template>

<script lang="ts" setup>
import { promptHandleKeyDown, promptHandleKeyUp } from "@/functions";
import { useSettings } from "@/store/settings";
import { useState } from "@/store/state";
import type { InferenceTabs } from "@/types";
import { NInput } from "naive-ui";
import type { PropType } from "vue";

const settings = useSettings();
const state = useState();

const props = defineProps({
  tab: {
    type: String as PropType<InferenceTabs>,
    required: true,
  },
});
</script>
