/**
 * Headless comparison renders for CYShape (museum-glass look).
 * Usage: NODE_PATH="$(npm root -g)" node grid.js
 */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = __dirname;
const SAMPLES = [
    [171, 156, 30],
    [104, 98, 12],
    [58, 55, 6],
    [247, 222, 50],
    [252, 251, 2],
    [284, 263, 42],
    [97, 77, 40],
    [184, 221, -74],
];

const HTML = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body { margin: 0; background: #0c1016; }
    canvas { display: block; width: 640px; height: 480px; }
  </style>
</head>
<body>
  <canvas id="c" width="640" height="480"></canvas>
  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script src="./static/js/cy-shape.js"></script>
  <script>
    window.renderSample = function (h11, h21, chi) {
      const canvas = document.getElementById('c');
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
      renderer.setSize(640, 480, false);
      renderer.setPixelRatio(1);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 640 / 480, 0.1, 100);
      CYShape.configureScene(THREE, { renderer, scene });

      const params = CYShape.paramsFor(h11, h21, chi);
      const geometry = CYShape.geometryFor(THREE, params, 240, 100);
      const material = CYShape.materialFor(THREE, params);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.set(-0.42, 0.55, 0.1);
      CYShape.frameMesh(camera, mesh, geometry);

      scene.add(mesh);
      renderer.render(scene, camera);
      return true;
    };
  </script>
</body>
</html>`;

async function main() {
    const htmlPath = path.join(ROOT, '_grid_preview.html');
    fs.writeFileSync(htmlPath, HTML);

    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
    await page.goto('file://' + htmlPath, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.THREE && window.CYShape);

    for (const [h11, h21, chi] of SAMPLES) {
        await page.evaluate(([a, b, c]) => window.renderSample(a, b, c), [h11, h21, chi]);
        await page.waitForTimeout(80);
        const out = path.join(ROOT, `m_${h11}_${h21}_${chi}.png`);
        await page.locator('canvas').screenshot({ path: out });
        console.log('wrote', path.basename(out));
    }

    await browser.close();
    fs.unlinkSync(htmlPath);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
