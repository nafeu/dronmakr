import { test, expect } from "@playwright/test";
import {
  assertTestMode,
  completeOnboarding,
  uniqueFilesRoot,
  waitForBackend,
} from "./helpers/app";

test.describe("onboarding", () => {
  test.beforeAll(async ({ request }) => {
    await waitForBackend(request);
    await assertTestMode(request);
  });

  test("completes first-run storage setup", async ({ page, request }) => {
    const filesRoot = uniqueFilesRoot("onboarding");
    await completeOnboarding(page, filesRoot);

    const settings = await request.get("/api/settings");
    expect(settings.ok()).toBeTruthy();
    const body = await settings.json();
    expect(body.FILES_ROOT).toContain(filesRoot.split("/").pop() ?? filesRoot);

    const status = await request.get("/api/settings/config-status");
    expect(status.ok()).toBeTruthy();
    const cfg = await status.json();
    expect(cfg.pluginPathsConfigured).toBeTruthy();
  });
});
