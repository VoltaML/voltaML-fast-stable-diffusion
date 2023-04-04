import { webSocketUrl } from "@/env";
import {
  processWebSocket,
  type WebSocketMessage,
} from "@/websockets/websockets";
import { useWebSocket } from "@vueuse/core";
import { useMessage, useNotification } from "naive-ui";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { useState } from "../store/state";

export const useWebsocket = defineStore("websocket", () => {
  const notificationProvider = useNotification();
  const messageProvider = useMessage();
  const global = useState();

  const onConnectedCallbacks: (() => void)[] = [];
  const onRefreshCallbacks: (() => void)[] = [];

  const websocket = useWebSocket(`${webSocketUrl}/api/websockets/master`, {
    heartbeat: {
      message: "ping",
      interval: 30000,
    },
    immediate: false,
    onMessage: (ws: WebSocket, event: MessageEvent) => {
      if (event.data === "pong") {
        return;
      }

      const data = JSON.parse(event.data) as WebSocketMessage;
      if (data.type === "refresh_models") {
        console.log("Recieved refresh_models message");
        onRefreshCallbacks.forEach((callback) => callback());
        return;
      }
      processWebSocket(data, global, notificationProvider);
    },
    onConnected: () => {
      messageProvider.success("Connected to server");
      onConnectedCallbacks.forEach((callback) => callback());
    },
    onDisconnected: () => {
      messageProvider.error("Disconnected from server");
    },
  });

  function ws_text() {
    switch (readyState.value) {
      case "CLOSED":
        return "Closed";
      case "CONNECTING":
        return "Connecting";
      case "OPEN":
        return "Connected";
    }
  }

  function get_color() {
    switch (readyState.value) {
      case "CLOSED":
        return "error";
      case "CONNECTING":
        return "warning";
      case "OPEN":
        return "success";
    }
  }

  const readyState = ref(websocket.status);
  const loading = computed(() => readyState.value === "CONNECTING");
  const text = computed(() => ws_text());
  const color = computed(() => get_color());

  return {
    websocket,
    readyState,
    loading,
    text,
    ws_open: websocket.open,
    color,
    onConnectedCallbacks,
    onRefreshCallbacks,
  };
});
