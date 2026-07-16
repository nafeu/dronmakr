import { APIRequestContext, expect, Page } from "@playwright/test";

export const SURGE_MATCH = /surge\s*xt/i;

export async function waitForBackend(request: APIRequestContext): Promise<void> {
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    const res = await request.get("/api/health");
    if (res.ok()) {
      const body = await res.json();
      expect(body.ok).toBe(true);
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("backend /api/health never became ready");
}

export async function assertTestMode(request: APIRequestContext): Promise<void> {
  const res = await request.get("/api/health");
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  expect(body.testMode).toBe(true);
}

export function uniqueFilesRoot(prefix = "dronmakr-e2e"): string {
  const stamp = Date.now();
  if (process.platform === "win32") {
    return `C:\\Users\\Public\\${prefix}-${stamp}`;
  }
  return `/tmp/${prefix}-${stamp}`;
}

export async function completeOnboarding(page: Page, filesRoot: string): Promise<void> {
  await page.goto("/onboarding");
  await page.locator("#FILES_ROOT").fill(filesRoot);
  const pluginPaths = page.locator("#PLUGIN_PATHS");
  if ((await pluginPaths.inputValue()).trim() === "") {
    const defaults = await page.request.get("/api/settings/plugin-path-defaults");
    const data = await defaults.json();
    if (data.pluginPaths) {
      await pluginPaths.fill(data.pluginPaths);
    }
  }
  await page.locator("#saveBtn").click();
  await expect(page).toHaveURL(/\/$|\/auditionr/, { timeout: 60_000 });
}

export async function gotoAuditionr(page: Page): Promise<void> {
  await page.goto("/auditionr");
  await expect(page.locator("#generatr-form")).toBeVisible({ timeout: 60_000 });
}

export async function requireSurgeXt(request: APIRequestContext): Promise<{ pluginPath: string; label: string }> {
  const res = await request.get("/api/generatr/drone-plugin-list?role=instrument");
  expect(res.ok()).toBeTruthy();
  const payload = await res.json();
  const plugins: Array<{ label?: string; pluginPath?: string; plugin_path?: string; name?: string }> =
    payload.detected ?? payload.plugins ?? payload.items ?? [];
  const match = plugins.find((p) => {
    const label = (p.label ?? p.name ?? "").toString();
    return SURGE_MATCH.test(label);
  });
  if (!match) {
    const names = plugins.map((p) => p.label ?? p.name ?? p.pluginPath ?? "?").join(", ");
    throw new Error(`Surge XT not detected under PLUGIN_PATHS. Found: ${names || "(none)"}`);
  }
  return {
    label: (match.label ?? match.name ?? "Surge XT").toString(),
    pluginPath: (match.pluginPath ?? match.plugin_path ?? "").toString(),
  };
}

export async function selectDroneInstrumentPlugin(page: Page, label: string): Promise<void> {
  await page.locator("#generatr-kind").selectOption("drone");
  await page.locator("#generatr-drone-instrument-slot").click();
  const picker = page.locator("#generatr-drone-plugin-picker-modal");
  await expect(picker).toBeVisible({ timeout: 30_000 });
  const choice = picker.locator("button.generatr-drone-plugin-picker-choice", { hasText: label });
  await expect(choice.first()).toBeVisible({ timeout: 60_000 });
  await choice.first().click();
  await expect(picker).toHaveClass(/hidden/, { timeout: 15_000 });
}

export async function clickGenerate(page: Page): Promise<void> {
  await page.locator("#generatr-btn").click();
  await expect(page.locator("#generatr-busy-overlay")).toBeHidden({ timeout: 180_000 });
}

export async function waitForLatestExport(
  request: APIRequestContext,
  timeoutMs = 180_000
): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const res = await request.get("/api/auditionr/exports");
    if (res.ok()) {
      const data = await res.json();
      const files: unknown[] = data.files ?? [];
      for (const entry of files) {
        if (typeof entry === "string" && entry.endsWith(".wav")) {
          return entry;
        }
        if (entry && typeof entry === "object") {
          const path = (entry as { path?: string; name?: string }).path
            ?? (entry as { path?: string; name?: string }).name
            ?? "";
          if (path.endsWith(".wav")) {
            return path;
          }
        }
      }
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error("no export WAV appeared in /api/auditionr/exports");
}
