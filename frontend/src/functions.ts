import { serverUrl } from "./env";
import { useWebsocket } from "./store/websockets";

export function dimensionValidator(value: number) {
  return value % 8 === 0;
}

export async function startWebsocket(messageProvider: any) {
  const websocketState = useWebsocket();

  const timeout = 1000;

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const response = await fetch(`${serverUrl}/api/test/alive`, {
    signal: controller.signal,
  }).catch(() => {
    messageProvider.error("Server is not responding");
  });
  clearTimeout(id);

  if (response === undefined) {
    return;
  }

  if (response.status !== 200) {
    messageProvider.error("Server is not responding");
    return;
  }

  console.log("Starting websocket");
  websocketState.ws_open();
}
