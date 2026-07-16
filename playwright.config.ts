import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:3766";
const isLinux = process.platform === "linux";

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: isLinux ? 300_000 : 180_000,
  expect: { timeout: isLinux ? 120_000 : 60_000 },
  reporter: [
    ["list"],
    ["html", { open: "never", outputFolder: process.env.PLAYWRIGHT_HTML_REPORT ?? "playwright-report" }],
  ],
  outputDir: process.env.E2E_ARTIFACTS_DIR ?? "test-results",
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    actionTimeout: isLinux ? 45_000 : 20_000,
    navigationTimeout: isLinux ? 60_000 : 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
