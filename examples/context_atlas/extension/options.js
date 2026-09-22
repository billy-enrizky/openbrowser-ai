const provider = document.getElementById("provider");
const apiKey = document.getElementById("api-key");
const status = document.getElementById("status");
const saveButton = document.getElementById("save");
const clearButton = document.getElementById("clear");

function sendMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!response?.ok) {
        reject(new Error(response?.error || "The extension request failed."));
        return;
      }
      resolve(response.payload || {});
    });
  });
}

async function refresh() {
  try {
    const selected = await sendMessage({ type: "context_atlas.provider_status" });
    provider.textContent = selected.provider === "jev"
      ? "Cloud is selected."
      : "Local is selected.";
    const keyStatus = await sendMessage({ type: "context_atlas.key_status" });
    clearButton.disabled = keyStatus.configured !== true;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not load settings.";
    status.dataset.kind = "error";
  }
}

document.getElementById("cloud-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const cleanKey = apiKey.value.trim();
  if (!cleanKey) {
    status.textContent = "Enter a Cloud access key before saving.";
    status.dataset.kind = "error";
    return;
  }
  saveButton.disabled = true;
  try {
    await sendMessage({ type: "context_atlas.save_key", api_key: cleanKey });
    apiKey.value = "";
    clearButton.disabled = false;
    status.textContent = "Cloud access key saved locally in this extension.";
    status.dataset.kind = "success";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not save the access key.";
    status.dataset.kind = "error";
  } finally {
    saveButton.disabled = false;
  }
});

clearButton.addEventListener("click", async () => {
  clearButton.disabled = true;
  try {
    await sendMessage({ type: "context_atlas.clear_key" });
    status.textContent = "Saved access key cleared.";
    status.dataset.kind = "success";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not clear the access key.";
    status.dataset.kind = "error";
    clearButton.disabled = false;
  }
});

void refresh();
