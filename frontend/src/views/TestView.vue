<template>
  <ModelPopup :model="model" />
</template>

<script setup lang="ts">
import ModelPopup from "@/components/models/ModelPopup.vue";
import { ref } from "vue";
import type { ICivitAIModel } from "../civitai";

const model = ref<ICivitAIModel | null>(null);

fetch("https://civitai.com/api/v1/models/7240").then((res) => {
  res.json().then((data) => {
    model.value = data;
  });
});
</script>
