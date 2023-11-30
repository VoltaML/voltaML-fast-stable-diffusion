<template>
  <div
    style="height: 500px; cursor: pointer"
    @click="emit('imgClick', props.column_index, props.item_index)"
  >
    <div
      :style="{
        filter:
          nsfwIndex(props.item.modelVersions[0].images[0].nsfw) >
            settings.data.settings.frontend.nsfw_ok_threshold && !filterOverride
            ? 'blur(32px)'
            : 'none',
        width: '100%',
        height: '100%',
      }"
    >
      <img
        v-if="props.item.modelVersions[0].images[0]?.url"
        :src="props.item.modelVersions[0].images[0].url"
        :style="{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
        }"
      />
    </div>
    <div
      style="
        position: absolute;
        width: 100%;
        bottom: 0;
        padding: 0 8px 0 12px;
        min-height: 32px;
        overflow: hidden;
        box-sizing: border-box;
        backdrop-filter: blur(12px);
        background-color: rgba(0, 0, 0, 0.3);
      "
    >
      <NText :depth="2">
        {{ item.name }}
      </NText>
    </div>
    <NButton
      v-if="
        nsfwIndex(props.item.modelVersions[0].images[0].nsfw) >
        settings.data.settings.frontend.nsfw_ok_threshold
      "
      type="error"
      style="
        position: absolute;
        top: 0;
        right: 0px;
        padding: 0 16px 0 12px;
        overflow: hidden;
        box-sizing: border-box;
        border-radius: 0px 0px 0px 8px;
      "
      @click.stop="filterOverride = !filterOverride"
    >
      <NIcon color="white" size="18">
        <EyeOffOutline />
      </NIcon>
    </NButton>
  </div>
</template>

<script setup lang="ts">
import { EyeOffOutline } from "@vicons/ionicons5";
import { NButton, NIcon, NText } from "naive-ui";
import type { PropType } from "vue";
import { ref } from "vue";
import type { ICivitAIModel } from "../../civitai";
import { nsfwIndex } from "../../civitai";
import { useSettings } from "../../store/settings";

const settings = useSettings();
const filterOverride = ref(false);

const props = defineProps({
  item: {
    type: Object as PropType<ICivitAIModel>,
    required: true,
  },
  column_index: {
    type: Number,
    required: true,
  },
  item_index: {
    type: Number,
    required: true,
  },
});
const emit = defineEmits(["imgClick"]);
</script>
