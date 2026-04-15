import { test, expect } from '@playwright/test';

test.describe('Three-column layout', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('main');
  });

  test('renders navbar, left panel, center, and right panel', async ({ page }) => {
    await expect(page.locator('.navbar')).toBeVisible();
    const main = page.locator('main');
    const children = main.locator('> *');
    await expect(children).toHaveCount(3);
  });

  test('left panel is scrollable at small viewport', async ({ page }) => {
    // Use a short viewport so content overflows
    await page.setViewportSize({ width: 1280, height: 500 });
    await page.waitForTimeout(100);

    const leftPanel = page.locator('main > aside').first();

    const overflowY = await leftPanel.evaluate(el => getComputedStyle(el).overflowY);
    expect(['auto', 'scroll']).toContain(overflowY);

    const { scrollHeight, clientHeight } = await leftPanel.evaluate(el => ({
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
    }));
    // At 500px viewport minus 64px navbar = 436px panel height.
    // Content is ~600px+ so it should overflow.
    expect(scrollHeight).toBeGreaterThan(clientHeight);

    // Actually scroll and verify
    await leftPanel.evaluate(el => el.scrollTo(0, 200));
    const scrollTop = await leftPanel.evaluate(el => el.scrollTop);
    expect(scrollTop).toBeGreaterThan(0);
  });

  test('left panel height is constrained to viewport (not expanding)', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 500 });
    await page.waitForTimeout(100);

    const leftPanel = page.locator('main > aside').first();
    const viewportHeight = 500;
    const box = await leftPanel.boundingBox();
    expect(box).toBeTruthy();
    // Panel must not exceed viewport height
    expect(box!.height).toBeLessThanOrEqual(viewportHeight);
    // And should be reasonable (viewport minus navbar ~64px)
    expect(box!.height).toBeGreaterThan(300);
    expect(box!.height).toBeLessThan(viewportHeight - 50);
  });

  test('scrolling left panel does not scroll center or right panel', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 500 });
    await page.waitForTimeout(100);

    const leftPanel = page.locator('main > aside').first();
    const centerPanel = page.locator('main > section');
    const rightPanel = page.locator('main > aside').last();

    const centerBefore = await centerPanel.evaluate(el => el.scrollTop);
    const rightBefore = await rightPanel.evaluate(el => el.scrollTop);

    await leftPanel.evaluate(el => el.scrollTo(0, 300));

    const centerAfter = await centerPanel.evaluate(el => el.scrollTop);
    const rightAfter = await rightPanel.evaluate(el => el.scrollTop);

    expect(centerAfter).toBe(centerBefore);
    expect(rightAfter).toBe(rightBefore);
  });

  test('grid columns have correct widths', async ({ page }) => {
    const leftPanel = page.locator('main > aside').first();
    const rightPanel = page.locator('main > aside').last();

    const leftBox = await leftPanel.boundingBox();
    const rightBox = await rightPanel.boundingBox();

    // Left = 18rem = 288px, right = 16rem = 256px
    expect(leftBox!.width).toBeGreaterThan(250);
    expect(leftBox!.width).toBeLessThan(320);
    expect(rightBox!.width).toBeGreaterThan(220);
    expect(rightBox!.width).toBeLessThan(290);
  });

  test('grid row constrains children to container height', async ({ page }) => {
    const main = page.locator('main');
    const leftPanel = page.locator('main > aside').first();

    const mainBox = await main.boundingBox();
    const leftBox = await leftPanel.boundingBox();

    // Left panel height should match main height (grid row = 1fr)
    expect(leftBox!.height).toBe(mainBox!.height);
  });
});

test.describe('Collapse sections', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('main');
  });

  test('collapse sections do not overlap each other', async ({ page }) => {
    const leftPanel = page.locator('main > aside').first();
    const collapses = leftPanel.locator('.collapse');
    const count = await collapses.count();
    expect(count).toBeGreaterThanOrEqual(3);

    for (let i = 0; i < count - 1; i++) {
      const current = await collapses.nth(i).boundingBox();
      const next = await collapses.nth(i + 1).boundingBox();
      if (current && next) {
        expect(current.y + current.height).toBeLessThanOrEqual(next.y + 1);
      }
    }
  });

  test('collapsing a section reduces content height', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 500 });
    await page.waitForTimeout(100);

    const leftPanel = page.locator('main > aside').first();
    const scrollHeightBefore = await leftPanel.evaluate(el => el.scrollHeight);

    // Close MPPI Parameters (3rd collapse)
    const mppiCollapse = leftPanel.locator('.collapse').nth(2);
    const checkbox = mppiCollapse.locator('input[type="checkbox"]');
    await checkbox.uncheck();
    await page.waitForTimeout(200);

    const scrollHeightAfter = await leftPanel.evaluate(el => el.scrollHeight);
    expect(scrollHeightAfter).toBeLessThan(scrollHeightBefore);
  });
});

test.describe('Floating toolbar', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('main');
  });

  test('toolbar buttons are visible', async ({ page }) => {
    await expect(page.locator('button', { hasText: 'Plan & Go' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'Play' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'Clear' })).toBeVisible();
  });

  test('toolbar is at bottom center of viewport', async ({ page }) => {
    const toolbar = page.locator('button', { hasText: 'Plan & Go' }).locator('..');
    const box = await toolbar.boundingBox();
    const viewport = page.viewportSize()!;

    expect(box).toBeTruthy();
    expect(box!.y + box!.height).toBeGreaterThan(viewport.height - 80);
    const centerX = box!.x + box!.width / 2;
    expect(centerX).toBeGreaterThan(viewport.width * 0.3);
    expect(centerX).toBeLessThan(viewport.width * 0.7);
  });

  test('toolbar is draggable', async ({ page }) => {
    const toolbar = page.locator('button', { hasText: 'Plan & Go' }).locator('..');
    const boxBefore = await toolbar.boundingBox();

    // Drag from the status text span
    const statusSpan = toolbar.locator('span').first();
    const spanBox = await statusSpan.boundingBox();
    if (spanBox) {
      await page.mouse.move(spanBox.x + 10, spanBox.y + 5);
      await page.mouse.down();
      await page.mouse.move(spanBox.x + 110, spanBox.y - 45, { steps: 5 });
      await page.mouse.up();

      const boxAfter = await toolbar.boundingBox();
      const moved = Math.abs(boxAfter!.x - boxBefore!.x) + Math.abs(boxAfter!.y - boxBefore!.y);
      expect(moved).toBeGreaterThan(20);
    }
  });
});

test.describe('Drawing tools', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('main');
  });

  test('5 drawing tool buttons exist', async ({ page }) => {
    const toolButtons = page.locator('.join button');
    await expect(toolButtons).toHaveCount(5);
  });

  test('clicking a tool highlights it', async ({ page }) => {
    const toolButtons = page.locator('.join button');
    await toolButtons.nth(1).click();
    const classes = await toolButtons.nth(1).getAttribute('class');
    expect(classes).toContain('btn-primary');
  });
});

test.describe('Presets', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('main');
  });

  test('all 4 preset buttons exist', async ({ page }) => {
    await expect(page.locator('button', { hasText: 'Straight' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'Lane Change' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'S-Curve' })).toBeVisible();
    await expect(page.locator('button', { hasText: 'Slalom' })).toBeVisible();
  });

  test('clicking preset updates status', async ({ page }) => {
    await page.locator('button', { hasText: 'Straight' }).click();
    const toolbar = page.locator('button', { hasText: 'Plan & Go' }).locator('..');
    await expect(toolbar).toContainText('Preset loaded');
  });
});
