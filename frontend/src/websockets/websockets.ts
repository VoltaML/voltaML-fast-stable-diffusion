export interface NotificationMessage {
  severity: "success" | "info" | "warning" | "error";
  title: string;
  timestamp: string;
  message: string;
  timeout: number;
  id: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
}

export function processWebSocket(message: WebSocketMessage): void {
  if (message.type === "test") {
    console.log(message.data);
  }
}
