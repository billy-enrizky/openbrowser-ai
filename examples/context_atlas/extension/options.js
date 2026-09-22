const provider = document.getElementById("provider");
const apiKey = document.getElementById("api-key");
const status = document.getElementById("status");
const saveButton = document.getElementById("save");
const clearButton = document.getElementById("clear");
let mutationInFlight = false;
let keyConfigured = false;

function updateMutationControls() {
  saveButton.disabled = mutationInFlight;
  clearButton.disabled = mutationInFlight || !keyConfigured;
}

function setMutationBusy(busy) {
  mutationInFlight = busy;
  updateMutationControls();
}

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

async function refresh({ preserveStatus = false } = {}) {
  try {
    const selected = await sendMessage({ type: "context_atlas.provider_status" });
    provider.textContent = selected.provider === "jev"
      ? "Cloud is selected."
      : "Local is selected.";
    const keyStatus = await sendMessage({ type: "context_atlas.key_status" });
    keyConfigured = keyStatus.configured === true;
    updateMutationControls();
  } catch (error) {
    if (preserveStatus) {
      console.warn("Could not refresh settings after a change.", error);
    } else {
      status.textContent = error instanceof Error ? error.message : "Could not load settings.";
      status.dataset.kind = "error";
    }
  }
}

document.getElementById("cloud-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (mutationInFlight) return;
  const cleanKey = apiKey.value.trim();
  if (!cleanKey) {
    status.textContent = "Enter a Cloud access key before saving.";
    status.dataset.kind = "error";
    return;
  }
  setMutationBusy(true);
  try {
    await sendMessage({ type: "context_atlas.save_key", api_key: cleanKey });
    apiKey.value = "";
    keyConfigured = true;
    status.textContent = "Cloud access key saved locally in this extension.";
    status.dataset.kind = "success";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not save the access key.";
    status.dataset.kind = "error";
  } finally {
    try {
      await refresh({ preserveStatus: true });
    } finally {
      setMutationBusy(false);
    }
  }
});

clearButton.addEventListener("click", async () => {
  if (mutationInFlight) return;
  setMutationBusy(true);
  try {
    await sendMessage({ type: "context_atlas.clear_key" });
    keyConfigured = false;
    status.textContent = "Saved access key cleared.";
    status.dataset.kind = "success";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not clear the access key.";
    status.dataset.kind = "error";
  }
  finally {
    try {
      await refresh({ preserveStatus: true });
    } finally {
      setMutationBusy(false);
    }
  }
});

void refresh();
