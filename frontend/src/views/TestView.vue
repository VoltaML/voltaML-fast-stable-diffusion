<template>
  <ModelPopup
    :model="model"
    :show-modal="showModal"
    @update:show-modal="(e) => (showModal = e)"
  />
</template>

<script setup lang="ts">
import ModelPopup from "@/components/models/ModelPopup.vue";
import { ref } from "vue";
import type { ICivitAIModel } from "../civitai";

const model = ref<ICivitAIModel | null>(null);
const showModal = ref(false);

fetch("https://civitai.com/api/v1/models/7240").then((res) => {
  res.json().then((data) => {
    model.value = data;
  });
});
</script>
