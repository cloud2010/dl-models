// Local browser checks through Chrome DevTools Protocol. No npm packages needed.
// Launch a local headless Chrome on port 9229 before running this script.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const destination = path.join(root, ".preview");
await fs.mkdir(destination, { recursive: true });
const targets = await (await fetch("http://127.0.0.1:9229/json/list")).json();
const target = targets.find(item => item.type === "page");
assert(target, "No Chrome page target found");
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map();
const errors = [];
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails);
  const task = pending.get(message.id);
  if (task) { pending.delete(message.id); message.error ? task.reject(message.error) : task.resolve(message.result); }
};
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function screenshot(name, full = false) {
  const options = { format: "png", captureBeyondViewport: full };
  if (full) {
    const layout = await send("Page.getLayoutMetrics");
    options.clip = { x: 0, y: 0, width: layout.cssContentSize.width, height: layout.cssContentSize.height, scale: 1 };
  }
  const result = await send("Page.captureScreenshot", options);
  await fs.writeFile(path.join(destination, name), Buffer.from(result.data, "base64"));
}
try {
  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: pathToFileURL(path.join(root, "index.html")).href });
  await evaluate("new Promise(resolve => { const ready = () => document.readyState === 'complete' ? resolve(true) : setTimeout(ready, 30); ready(); })");
  await evaluate("Promise.all([...document.images].filter(img=>img.getAttribute('src')).map(img=>{img.loading='eager';return img.decode();}))");
  const summary = await evaluate(`({title:document.title, timeline:document.querySelectorAll('.timeline-item').length,
    photos:document.querySelectorAll('.photo-card').length, configurations:document.querySelectorAll('#configuration option').length,
    images:[...document.images].filter(img=>img.getAttribute('src')).map(img=>({src:img.getAttribute('src'),loaded:img.complete&&img.naturalWidth>0}))})`);
  assert.equal(summary.timeline, 5); assert.equal(summary.photos, 4); assert.equal(summary.configurations, 8);
  assert(summary.images.every(image => image.loaded), JSON.stringify(summary.images));
  await evaluate("document.querySelector('[data-filter=deep]').click()");
  assert.equal(await evaluate("document.querySelectorAll('.model-note:not([hidden])').length"), 2);
  await evaluate("document.querySelector('[data-filter=all]').click()");
  await evaluate("document.querySelector('.photo-card button').click()");
  assert.equal(await evaluate("document.querySelector('#photo-dialog').open"), true);
  await evaluate("document.querySelector('#next-photo').click()");
  assert.match(await evaluate("document.querySelector('#photo-position').textContent"), /02 OF 04/);
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 });
  assert.equal(await evaluate("document.querySelector('#photo-dialog').open"), false);
  await evaluate("document.querySelector('[data-metric=acc]').click(); document.querySelector('#configuration').value='3'; document.querySelector('#configuration').dispatchEvent(new Event('change')); document.querySelector('#feature-slider').value='941'; document.querySelector('#feature-slider').dispatchEvent(new Event('input'))");
  assert.equal(await evaluate("document.querySelector('#value-readout').textContent"), await evaluate("ALBUM_DATA.configurations[3].acc[940].toFixed(6)"));
  const exported = await evaluate(`(async () => {
    const link=document.querySelector('#save-chart');
    link.addEventListener('click',event=>event.preventDefault(),{once:true}); link.click();
    const text=await (await fetch(link.href)).text();
    const svgDocument=new DOMParser().parseFromString(text,'image/svg+xml');
    return {filename:link.download,errors:svgDocument.querySelectorAll('parsererror').length,text};
  })()`);
  assert.equal(exported.errors, 0);
  assert.match(exported.filename, /-acc\.svg$/);
  assert.match(exported.text, /Historical log, not revalidated/);
  await evaluate("document.querySelector('[data-metric=mcc]').click(); document.querySelector('#configuration').value='0'; document.querySelector('#configuration').dispatchEvent(new Event('change')); document.querySelector('#feature-slider').value='400'; document.querySelector('#feature-slider').dispatchEvent(new Event('input')); document.activeElement.blur(); window.scrollTo({top:0,behavior:'instant'})");
  await screenshot("album-desktop.png", true);
  await screenshot("album-cover.png");
  const viewports = [];
  for (const width of [768, 390, 340]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height: 844, deviceScaleFactor: 1, mobile: true });
    await evaluate("window.scrollTo({top:0,behavior:'instant'})");
    const layout = await evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth})");
    assert(layout.scroll <= width + 1, `Horizontal overflow at ${width}: ${JSON.stringify(layout)}`);
    viewports.push(layout);
    if (width === 390) { await screenshot("album-mobile.png", true); await screenshot("album-mobile-cover.png"); }
  }
  assert.equal(errors.length, 0, JSON.stringify(errors));
  const report = { summary, viewports, javascriptErrors: errors.length, checks: ["original images", "timeline", "model filtering", "photo dialog", "photo navigation", "Escape closes dialog", "curve matches source", "valid current SVG export", "responsive overflow"] };
  await fs.writeFile(path.join(destination, "browser-check.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally { socket.close(); }
