import { webSocketUrl } from "@/env";
import {
  processWebSocket,
  type WebSocketMessage,
} from "@/websockets/websockets";
import { useWebSocket } from "@vueuse/core";
import { useNotification } from "naive-ui";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { useState } from "../store/state";

export const useWebsocket = defineStore("websocket", () => {
  const notificationProvider = useNotification();
  const global = useState();
  const websocket = useWebSocket(`${webSocketUrl}/api/websockets/master`, {
    autoReconnect: {
      delay: 3000,
    },
    heartbeat: {
      message: "ping",
      interval: 30000,
    },
    onMessage: (ws: WebSocket, event: MessageEvent) => {
      console.info(event.data);
      if (event.data === "pong") {
        return;
      }
      const data = JSON.parse(event.data) as WebSocketMessage;
      processWebSocket(data, global, notificationProvider);
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
  };
});
