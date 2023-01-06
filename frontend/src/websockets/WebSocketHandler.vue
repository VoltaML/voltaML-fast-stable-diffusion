<template>
  <slot />
</template>

<script setup lang="ts">
import { useWebsocket } from "@/store/websockets";
import { useNotification } from "naive-ui";

const notification = useNotification();
const ws = useWebsocket();

ws.injectNotificationHanler(notification);
</script>
