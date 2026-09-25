import { chromium } from "playwright";

const BASE = "http://127.0.0.1:3000";
const OUT = process.env.SHOT_DIR;
const errors = [];

const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium" });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

const step = async (name, fn) => {
  process.stdout.write(`  ${name} ... `);
  try { await fn(); console.log("ok"); }
  catch (e) { console.log("FAILED: " + e.message.split("\n")[0]); throw e; }
};

await step("signup", async () => {
  await page.goto(`${BASE}/signup`, { waitUntil: "networkidle" });
  await page.fill('input[name="organization_name"]', "Acme Plumbing");
  await page.fill('input[name="email"]', "owner@acme-plumbing.co");
  await page.fill('input[name="password"]', "correct-horse-staple-9");
  await page.click('button[type="submit"]');
  await page.waitForURL("**/dashboard", { timeout: 20000 });
});

await step("dashboard renders quota meters", async () => {
  await page.waitForSelector('[role="meter"]', { timeout: 15000 });
  const meters = await page.locator('[role="meter"]').count();
  if (meters < 4) throw new Error(`expected 4 meters, saw ${meters}`);
});
await page.screenshot({ path: `${OUT}/01-dashboard.png`, fullPage: true });

await step("start an audit", async () => {
  await page.fill('input[name="url"]', "http://127.0.0.1:8099/index.html");
  await page.fill('input[name="max_pages"]', "6");
  await page.click('button:has-text("Start audit")');
  await page.waitForSelector('text=/Audit queued/', { timeout: 20000 });
});

await step("audit completes", async () => {
  await page.waitForSelector('text=complete', { timeout: 60000 });
});
await page.screenshot({ path: `${OUT}/02-audit-done.png`, fullPage: true });

await step("open the audit", async () => {
  await page.click('a[href^="/audits/"]');
  await page.waitForURL("**/audits/**", { timeout: 20000 });
  await page.waitForSelector("text=Overall GEO score", { timeout: 20000 });
  // The detail page polls while the worker runs; wait for it to settle rather
  // than asserting against a half-finished audit.
  await page.waitForSelector('text="What to change"', { timeout: 60000 });
});
await page.screenshot({ path: `${OUT}/03-audit-detail.png`, fullPage: true });

await step("signal breakdown + recommendations present", async () => {
  const body = await page.textContent("body");
  for (const needle of ["Structured data", "Heading hierarchy", "What to change"]) {
    if (!body.includes(needle)) throw new Error(`missing "${needle}"`);
  }
});

await step("data table view opens", async () => {
  await page.locator('button:has-text("Show data table")').first().click();
  await page.waitForSelector("table", { timeout: 5000 });
});

await step("settings: create an API key", async () => {
  await page.click('a[href="/settings"]');
  await page.waitForURL("**/settings", { timeout: 20000 });
  await page.fill('input[name="name"]', "CI pipeline");
  await page.click('button:has-text("Create key")');
  await page.waitForSelector("text=/gk_live_/", { timeout: 20000 });
});
await page.screenshot({ path: `${OUT}/04-settings.png`, fullPage: true });

await step("quota refusal is shown to the user", async () => {
  // Free plan allows one API key; a second must be refused with a clear message.
  await page.fill('input[name="name"]', "second key");
  await page.click('button:has-text("Create key")');
  await page.waitForSelector("text=/active API keys/", { timeout: 20000 });
});

await step("dark mode renders", async () => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });
  await page.waitForSelector('[role="meter"]', { timeout: 15000 });
});
await page.screenshot({ path: `${OUT}/05-dark.png`, fullPage: true });

await step("mobile width has no horizontal scroll", async () => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload({ waitUntil: "networkidle" });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 1,
  );
  if (overflow) throw new Error("page scrolls horizontally at 390px");
});
await page.screenshot({ path: `${OUT}/06-mobile.png`, fullPage: true });

await step("sign out clears the session", async () => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle" });
  await page.click('button:has-text("Sign out")');
  await page.waitForURL("**/login", { timeout: 20000 });
  await page.goto(`${BASE}/dashboard`);
  await page.waitForURL("**/login", { timeout: 20000 });
});

await browser.close();
console.log(errors.length ? `\nCONSOLE ERRORS:\n${errors.join("\n")}` : "\nno console errors");
