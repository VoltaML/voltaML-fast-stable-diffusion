const loc = window.location;
let new_uri;
if (loc.protocol === "https:") {
  new_uri = "wss:";
} else {
  new_uri = "ws:";
}

export const serverUrl = import.meta.env.DEV
  ? "http://localhost:5003"
  : loc.protocol + "//" + loc.host;
export const webSocketUrl = new_uri + "//" + loc.host;
