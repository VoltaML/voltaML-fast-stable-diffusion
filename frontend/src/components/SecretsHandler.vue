<template>
  <NModal
    :show="global.state.secrets.huggingface !== 'ok'"
    preset="card"
    title="Missing HuggingFace Token"
    style="width: 80vw"
    :closable="false"
  >
    <NText>
      API does not have a HuggingFace token. Please enter a valid token to
      continue. You can get a token from
      <a target="_blank" href="https://huggingface.co/settings/tokens"
        >this page</a
      >
    </NText>
    <NInput
      type="password"
      placeholder="hf_123..."
      style="margin-top: 8px"
      :allow-input="noSideSpace"
      v-model:value="hf_token"
    >
    </NInput>
    <div
      style="margin-top: 8px; width: 100%; display: flex; justify-content: end"
    >
      <NButton
        ghost
        type="primary"
        :loading="hf_loading"
        @click="setHuggingfaceToken"
        >Set Token</NButton
      >
    </div>
  </NModal>
</template>

<script lang="ts" setup>
import { serverUrl } from "@/env";
import { useState } from "@/store/state";
import { NButton, NInput, NModal, NText, useMessage } from "naive-ui";
import { ref } from "vue";

const message = useMessage();
const global = useState();

const hf_loading = ref(false);
const hf_token = ref("");
function noSideSpace(value: string) {
  return !/ /g.test(value);
}

function setHuggingfaceToken() {
  hf_loading.value = true;

  const url = new URL(`${serverUrl}/api/settings/inject-var-into-dotenv`);
  url.searchParams.append("key", "HUGGINGFACE_TOKEN");
  url.searchParams.append("value", hf_token.value);
  fetch(url, { method: "POST" })
    .then((res) => {
      if (res.status !== 200) {
        message.create("Failed to set HuggingFace token", { type: "error" });
        return;
      }
      global.state.secrets.huggingface = "ok";
      message.create("HuggingFace token set successfully", { type: "success" });
    })
    .catch((e: Error) => {
      message.create(`Failed to set HuggingFace token: ${e.message}`, {
        type: "error",
      });
    });
  hf_loading.value = false;
}
</script>
