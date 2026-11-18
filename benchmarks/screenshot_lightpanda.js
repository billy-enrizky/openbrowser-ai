async function captureProof({ url, nonce }) {
  const page = new Page();
  await page.goto(url, { waitUntil: 'load', timeout: 15000 });

  const ready = await page.evaluate(`JSON.stringify({
    readyState: document.readyState,
    readyMarker: document.documentElement.dataset.screenshotReady || null,
    imageReady: (() => {
      const image = document.querySelector('#proof img');
      return Boolean(image && image.complete && image.naturalWidth === 64 && image.naturalHeight === 64);
    })(),
  })`);
  const observation = typeof ready === 'string' ? JSON.parse(ready) : ready;
  if (observation?.readyState !== 'complete' || observation?.readyMarker !== nonce || observation?.imageReady !== true) {
    throw new Error('screenshot fixture did not reach its ready marker');
  }

  await page.screenshot({ selector: '#proof', path: 'result.png', fullPage: false, timeout: 15000 });
}
