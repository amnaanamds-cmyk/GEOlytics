import { chromium } from "playwright";
const BASE = "http://127.0.0.1:3000";
const OUT = process.env.SHOT_DIR;
const errors = [];
const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium" });
const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(`${BASE}/login`, { waitUntil: "networkidle" });
await page.fill('input[name="email"]', "owner@acme-plumbing.co");
await page.fill('input[name="password"]', "correct-horse-staple-9");
await page.click('button[type="submit"]');
await page.waitForURL("**/dashboard", { timeout: 20000 });

await page.click('a[href="/experiments"]');
await page.waitForURL("**/experiments", { timeout: 20000 });
await page.click('a[href^="/experiments/"]');
await page.waitForURL("**/experiments/**", { timeout: 20000 });
await page.waitForSelector("text=95% CI", { timeout: 20000 });
console.log("delta plot rendered");

// The metric filter must refetch and repaint.
await page.selectOption('select >> nth=0', "recall@10");
await page.waitForSelector("text=Difference in recall@10", { timeout: 20000 });
console.log("metric filter refetches");

await page.screenshot({ path: `${OUT}/07-experiment.png`, fullPage: true });
await browser.close();
console.log(errors.length ? `CONSOLE ERRORS:\n${errors.join("\n")}` : "no console errors");
