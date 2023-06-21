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
export const webSocketUrl = import.meta.env.DEV
  ? "ws://localhost:5003"
  : new_uri + "//" + loc.host;
export const huggingfaceModelsFile = import.meta.env.DEV
  ? `${serverUrl}/api/test/huggingface-models.json`
  : "https://raw.githubusercontent.com/VoltaML/voltaML-fast-stable-diffusion/experimental/static/huggingface-models.json";
