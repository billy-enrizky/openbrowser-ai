const PROVIDER_ORIGINS = Object.freeze({
  jev: ["https://api.typesafe.ai/v1/systemone"],
  laya: [
    "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/d9d003d543e63d6d3375c21d44624136bd1e0bad/*",
    "https://us.aws.cdn.hf.co/xet-bridge-us/*",
  ],
});

const params = new URLSearchParams(location.search);
const provider = params.get("provider");
const requestId = params.get("request_id") || "";
const origins = PROVIDER_ORIGINS[provider];
const heading = document.getElementById("heading");
const description = document.getElementById("description");
const allowButton = document.getElementById("allow");
const closeButton = document.getElementById("close");
const status = document.getElementById("status");

if (provider === "jev") {
  heading.textContent = "Allow Cloud access";
  description.textContent = "Context Atlas uses Cloud access only when you choose Cloud search.";
} else if (provider === "laya") {
  heading.textContent = "Allow Local model access";
  description.textContent = "Context Atlas downloads the pinned Local model when you start Local search.";
} else {
  allowButton.disabled = true;
  status.textContent = "The selected provider is not supported.";
}

function sendMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!response?.ok) {
        reject(new Error(response?.error || "The access request could not be completed."));
        return;
      }
      resolve(response.payload || {});
    });
  });
}

allowButton.addEventListener("click", async () => {
  if (!origins) return;
  allowButton.disabled = true;
  closeButton.disabled = true;
  status.textContent = "Waiting for your permission…";
  let permissionPromise;
  try {
    permissionPromise = chrome.permissions.request({ origins });
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "The access request could not be started.";
    allowButton.disabled = false;
    closeButton.disabled = false;
    return;
  }
  try {
    const granted = await permissionPromise;
    const result = await sendMessage({
      type: "context_atlas.provider_access_result",
      provider,
      granted,
      request_id: requestId,
    });
    status.textContent = result.granted
      ? "Access granted. Return to your page and choose this provider again."
      : "Access was not granted. Return to your page and try again if you change your mind.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "The access request could not be completed.";
  } finally {
    allowButton.disabled = false;
    closeButton.disabled = false;
  }
});

closeButton.addEventListener("click", () => window.close());
