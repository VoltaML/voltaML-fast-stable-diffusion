import { serverUrl, webSocketUrl } from "@/env";
import {
  processWebSocket,
  type WebSocketMessage,
} from "@/websockets/websockets";
import { useWebSocket } from "@vueuse/core";
import { useNotification } from "naive-ui";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { useState } from "../store/state";
import { useSettings } from "./settings";

export const useWebsocket = defineStore("websocket", () => {
  const notificationProvider = useNotification();
  const global = useState();
  const conf = useSettings();
  const websocket = useWebSocket(`${webSocketUrl}/api/websockets/master`, {
    autoReconnect: {
      delay: 3000,
    },
    heartbeat: {
      message: "ping",
      interval: 30000,
    },
    onMessage: (ws: WebSocket, event: MessageEvent) => {
      if (event.data === "pong") {
        return;
      }
      console.info(event.data);
      const data = JSON.parse(event.data) as WebSocketMessage;
      processWebSocket(data, global, notificationProvider);
    },
    onConnected: () => {
      fetch(`${serverUrl}/api/models/loaded`).then((response) => {
        if (response.status === 200) {
          response.json().then((data) => {
            if (data.length === 0) {
              conf.data.settings.model = "none";
              return;
            }
            conf.data.settings.model = data[0][0];
          });
        }
      });
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
