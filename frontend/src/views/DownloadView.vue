<template>
  <NSpace
    justify="end"
    inline
    align="center"
    class="install"
    style="width: 100%; margin: 8px"
  >
    <NInput
      v-model:value="customModel"
      placeholder="Custom model"
      style="width: 350px"
    />
    <NButton
      type="primary"
      bordered
      @click="downloadModel"
      :loading="conf.state.downloading"
      :disabled="conf.state.downloading || customModel === ''"
      secondary
      style="margin-right: 16px"
      >Install</NButton
    >
  </NSpace>
  <div
    style="
      height: 50vh;
      display: inline-flex;
      justify-content: center;
      width: 100%;
    "
  >
    <NResult status="info" title="Download" description="Work in progress" />
  </div>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { NButton, NInput, NResult, NSpace } from "naive-ui";
import { ref } from "vue";
import { useState } from "../store/state";

const conf = useState();

const customModel = ref("");

function downloadModel() {
  const url = new URL(`${serverUrl}/api/models/download`);
  url.searchParams.append("model", customModel.value);
  console.log(url);
  conf.state.downloading = true;
  customModel.value = "";
  fetch(url, { method: "POST" })
    .then(() => {
      conf.state.downloading = false;
    })
    .catch(() => {
      conf.state.downloading = false;
    });
}
</script>
