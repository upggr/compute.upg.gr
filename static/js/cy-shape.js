/*
 * cy-shape.js — maps Calabi-Yau invariants to visual parameters.
 *
 * Design goal: two manifolds that differ topologically must LOOK different,
 * and the difference must be readable — presented as museum glass on a dark
 * plinth (crisp speculars, subtle metal, no HDR/bloom).
 *
 *   lobes      <- h11  (Kahler moduli)        more moduli  -> more lobes
 *   undulation <- h21  (complex structure)    more moduli  -> finer ripples
 *   asymmetry  <- chi  (Euler characteristic) sign of chi  -> which side swells
 *   girth      <- total moduli                bigger total -> chunkier tube
 *   hue        <- chi                         negative=cool, positive=warm
 *   saturation <- |chi| / total moduli        topological "extremeness"
 */
(function (global) {
    'use strict';

    var CLEAR_COLOR = 0x0c1016;

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function compress(value, midpoint) {
        const v = Math.max(0, Number(value) || 0);
        return clamp(Math.log1p(v) / Math.log1p(midpoint * 2), 0, 1);
    }

    function paramsFor(h11, h21, chi) {
        h11 = Math.max(0, Number(h11) || 0);
        h21 = Math.max(0, Number(h21) || 0);
        chi = Number(chi) || 0;

        const total = h11 + h21;
        const kahler = compress(h11, 150);
        const complex = compress(h21, 150);
        const size = compress(total, 300);
        const skew = total > 0 ? clamp(chi / total, -1, 1) : 0;

        const f1 = 3 + Math.round(kahler * 5);
        const f2 = 2 + Math.round(complex * 4);
        const f3 = 1 + Math.round(compress(Math.abs(chi), 200) * 3);
        const f4 = 1 + Math.round((kahler + complex) * 2);
        const f5 = 1 + Math.round(Math.abs(kahler - complex) * 5);

        return {
            a: 1.0 + size * 0.5,
            b: 0.34 + complex * 0.28,
            c: 0.1 + Math.abs(skew) * 0.35,
            f1: f1,
            f2: f2,
            f3: f3,
            f4: f4,
            f5: f5,
            // Slightly stronger warp so ridges catch the museum key light.
            warp: 0.08 + kahler * 0.22,
            twist: 0.05 + complex * 0.16,
            skew: skew,
            size: size
        };
    }

    // Hue: blue (chi < 0) -> teal (chi ~ 0) -> amber (chi > 0).
    function paletteFor(params) {
        const skew = params.skew || 0;
        const hue = (0.58 - skew * 0.42 + 1) % 1;
        // Keep enough saturation for chi encoding under metallic lighting.
        const saturation = clamp(0.42 + Math.abs(skew) * 0.4, 0.38, 0.82);
        return { hue: hue, saturation: saturation };
    }

    function applyVertexColors(THREE, geometry, params) {
        const palette = paletteFor(params);
        const position = geometry.attributes.position;
        const color = new THREE.Color();
        const colors = [];

        let maxRadius = 0;
        for (let i = 0; i < position.count; i++) {
            const r = Math.hypot(position.getX(i), position.getY(i));
            if (r > maxRadius) maxRadius = r;
        }
        maxRadius = maxRadius || 1;

        for (let i = 0; i < position.count; i++) {
            const x = position.getX(i);
            const y = position.getY(i);
            const z = position.getZ(i);
            const radius = Math.hypot(x, y) / maxRadius;
            const angle = Math.atan2(y, x);

            const band = 0.5 + 0.5 * Math.cos(angle * params.f1);
            const hue = (palette.hue + band * 0.05 + radius * 0.03) % 1;
            // Lifted value range: ridges catch speculars instead of reading chalky.
            const lightness = clamp(
                0.42 + 0.22 * radius + 0.12 * band + 0.1 * Math.tanh(z),
                0.28,
                0.72
            );
            // Desaturate midtones slightly; keep lobe bands punchier.
            const sat = clamp(palette.saturation * (0.85 + band * 0.25), 0.3, 0.85);
            color.setHSL(hue, sat, lightness);
            colors.push(color.r, color.g, color.b);
        }

        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        return geometry;
    }

    function materialFor(THREE, params) {
        const extremeness = Math.abs(params.skew || 0);
        return new THREE.MeshStandardMaterial({
            vertexColors: true,
            metalness: 0.55 + extremeness * 0.3,
            roughness: 0.4 - extremeness * 0.22,
            flatShading: false,
            emissive: 0x000000,
            emissiveIntensity: 0
        });
    }

    /**
     * Museum-plinth lighting: charcoal clear, warm key, cool rim, soft fill.
     * Returns the lights for optional later tweaks.
     */
    function configureScene(THREE, options) {
        const renderer = options.renderer;
        const scene = options.scene;
        const clear = options.clearColor != null ? options.clearColor : CLEAR_COLOR;

        renderer.setClearColor(clear, 1);

        const fill = new THREE.AmbientLight(0xb8c4d4, 0.28);
        scene.add(fill);

        const key = new THREE.DirectionalLight(0xfff0e0, 1.35);
        key.position.set(4.5, 7, 6);
        scene.add(key);

        const rim = new THREE.DirectionalLight(0x7eb6ff, 0.7);
        rim.position.set(-6, 2, -4);
        scene.add(rim);

        const bounce = new THREE.PointLight(0xa8b8cc, 0.45, 40);
        bounce.position.set(-2, -4, 5);
        scene.add(bounce);

        return { fill: fill, key: key, rim: rim, bounce: bounce };
    }

    /**
     * Center mesh on origin and pull camera back to fit the bounding sphere.
     */
    function frameMesh(camera, mesh, geometry, padding) {
        padding = padding == null ? 1.18 : padding;
        geometry.computeBoundingSphere();
        const bounds = geometry.boundingSphere;
        mesh.position.set(-bounds.center.x, -bounds.center.y, -bounds.center.z);
        const fitDistance = (bounds.radius / Math.sin((camera.fov * Math.PI / 180) / 2)) * padding;
        camera.position.set(0, 0, fitDistance);
        camera.near = Math.max(0.05, fitDistance / 100);
        camera.far = fitDistance * 4;
        camera.updateProjectionMatrix();
        return bounds;
    }

    function buildParametricGeometry(THREE, surface, slices, stacks) {
        const vertices = [];
        const indices = [];
        const uvs = [];
        const point = new THREE.Vector3();

        for (let i = 0; i <= stacks; i++) {
            const v = i / stacks;
            for (let j = 0; j <= slices; j++) {
                const u = j / slices;
                surface(u, v, point);
                vertices.push(point.x, point.y, point.z);
                uvs.push(u, v);
            }
        }

        const rowLength = slices + 1;
        for (let i = 0; i < stacks; i++) {
            for (let j = 0; j < slices; j++) {
                const a = i * rowLength + j;
                const b = a + rowLength;
                indices.push(a, b, a + 1);
                indices.push(b, b + 1, a + 1);
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setIndex(indices);
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        geometry.computeVertexNormals();
        return geometry;
    }

    function surfaceFor(params) {
        return function (u, v, target) {
            const U = u * Math.PI * 2;
            const V = v * Math.PI * 2;
            const lobes = params.warp * Math.cos(params.f1 * U);
            const tube = params.b * (1 + 0.18 * Math.sin(params.f4 * V));
            const radius = params.a + lobes + tube * Math.cos(V);
            const x = radius * Math.cos(U);
            const y = radius * Math.sin(U);
            const z = tube * Math.sin(V)
                + params.c * Math.sin(params.f3 * U)
                + params.twist * Math.sin(params.f5 * U + params.f2 * V);
            target.set(x, y, z);
        };
    }

    function geometryFor(THREE, params, slices, stacks) {
        const geometry = buildParametricGeometry(
            THREE, surfaceFor(params), slices || 220, stacks || 90);
        return applyVertexColors(THREE, geometry, params);
    }

    global.CYShape = {
        CLEAR_COLOR: CLEAR_COLOR,
        paramsFor: paramsFor,
        paletteFor: paletteFor,
        applyVertexColors: applyVertexColors,
        materialFor: materialFor,
        configureScene: configureScene,
        frameMesh: frameMesh,
        surfaceFor: surfaceFor,
        buildParametricGeometry: buildParametricGeometry,
        geometryFor: geometryFor
    };
})(window);
