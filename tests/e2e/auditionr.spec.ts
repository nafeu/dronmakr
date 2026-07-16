import { test, expect } from "@playwright/test";
import {
  assertTestMode,
  clickGenerate,
  completeOnboarding,
  gotoAuditionr,
  requireSurgeXt,
  selectDroneInstrumentPlugin,
  uniqueFilesRoot,
  waitForBackend,
  waitForLatestExport,
} from "./helpers/app";
import { validateWavFile } from "./helpers/audio";

test.describe("auditionr smoke", () => {
  let surge: { pluginPath: string; label: string };

  test.beforeAll(async ({ request }) => {
    await waitForBackend(request);
    await assertTestMode(request);
    surge = await requireSurgeXt(request);
  });

  test("onboarding → Surge XT → generate → valid WAV", async ({ page, request }) => {
    const filesRoot = uniqueFilesRoot("auditionr");
    await completeOnboarding(page, filesRoot);
    await gotoAuditionr(page);

    const listRes = await request.get("/api/generatr/drone-plugin-list?role=instrument");
    expect(listRes.ok()).toBeTruthy();

    await selectDroneInstrumentPlugin(page, surge.label);
    await clickGenerate(page);

    const exportPath = await waitForLatestExport(request);
    expect(exportPath.endsWith(".wav")).toBeTruthy();

    const validation = validateWavFile(exportPath);
    expect(validation.ok).toBe(true);
    expect(validation.durationS).toBeGreaterThan(0);
    expect(validation.peak).toBeGreaterThan(1e-5);
    expect(validation.rms).toBeGreaterThan(1e-5);
  });
});
