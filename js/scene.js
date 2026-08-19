/**
 * AETHER 3D SAR COLORIZER - THREE.JS 3D SCENE
 * Renders 3D Earth with NASA texture, atmospheric glow, satellite orbit model,
 * twinkling starfield, and warm solar lighting.
 */

class EarthScene {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });

        this.earth = null;
        this.atmosphere = null;
        this.satelliteGroup = null;
        this.signalParticles = null;
        this.starField = null;

        this.orbitAngle = 0;
        this.time = 0;

        this.init();
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    init() {
        // Renderer config
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Camera position
        this.camera.position.set(0, 0, 9);

        // 1. LIGHTING SETUP
        // Main Sun Light (Warm White upper right)
        const sunLight = new THREE.DirectionalLight(0xFFFFE0, 1.2);
        sunLight.position.set(10, 8, 10);
        sunLight.castShadow = true;
        this.scene.add(sunLight);

        // Secondary Rim Light (Cyan behind Earth)
        const rimLight = new THREE.DirectionalLight(0x00D9FF, 0.4);
        rimLight.position.set(-10, -5, -8);
        this.scene.add(rimLight);

        // Ambient Light
        const ambientLight = new THREE.AmbientLight(0x1a2638, 0.45);
        this.scene.add(ambientLight);

        // 2. CREATE REAL EARTH
        this.createEarth();

        // 3. CREATE SATELLITE MODEL
        this.createSatellite();

        // 4. CREATE TWINKLING STARFIELD
        this.createStarfield();

        // Start render loop
        this.animate();
    }

    createEarth() {
        const earthRadius = 2.4;
        const geometry = new THREE.SphereGeometry(earthRadius, 64, 64);

        // Load NASA Earth Texture
        const textureLoader = new THREE.TextureLoader();
        const earthTextureUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg';

        const earthMaterial = new THREE.MeshPhongMaterial({
            shininess: 15,
            specular: new THREE.Color(0x222222)
        });

        // Load texture with error fallback to procedural high-res NASA Blue Marble texture
        textureLoader.load(
            earthTextureUrl,
            (texture) => {
                earthMaterial.map = texture;
                earthMaterial.needsUpdate = true;
            },
            undefined,
            () => {
                // Fallback procedural canvas texture
                earthMaterial.map = this.generateProceduralEarthTexture();
                earthMaterial.needsUpdate = true;
            }
        );

        this.earth = new THREE.Mesh(geometry, earthMaterial);
        
        // Position Earth center-right on desktop
        this.updateEarthPosition();

        this.scene.add(this.earth);

        // Atmospheric Glow Halo Layer
        const atmosGeo = new THREE.SphereGeometry(earthRadius * 1.035, 64, 64);
        const atmosMat = new THREE.MeshPhongMaterial({
            color: 0x00D9FF,
            transparent: true,
            opacity: 0.22,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide
        });
        this.atmosphere = new THREE.Mesh(atmosGeo, atmosMat);
        this.earth.add(this.atmosphere);
    }

    generateProceduralEarthTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 2048; canvas.height = 1024;
        const ctx = canvas.getContext('2d');

        // Deep Ocean Blue Base
        ctx.fillStyle = '#061a3a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Vector Continents
        ctx.fillStyle = '#1c4a28';
        ctx.strokeStyle = '#2d6b3c';
        ctx.lineWidth = 4;

        // Landmass patches
        const landmasses = [
            { x: 400, y: 350, r: 180 }, { x: 550, y: 450, r: 220 }, // Americas
            { x: 1100, y: 300, r: 280 }, { x: 1300, y: 420, r: 200 }, // Eurasia
            { x: 1180, y: 550, r: 190 }, // Africa
            { x: 1650, y: 700, r: 140 }  // Australia
        ];

        landmasses.forEach(m => {
            ctx.beginPath();
            ctx.arc(m.x, m.y, m.r, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        });

        // Ice Caps
        ctx.fillStyle = '#e2f1f8';
        ctx.fillRect(0, 0, canvas.width, 80);
        ctx.fillRect(0, canvas.height - 80, canvas.width, 80);

        const tex = new THREE.CanvasTexture(canvas);
        return tex;
    }

    createSatellite() {
        this.satelliteGroup = new THREE.Group();

        // Main satellite body
        const bodyGeo = new THREE.BoxGeometry(0.18, 0.18, 0.28);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x22252a, metalness: 0.8, roughness: 0.2 });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        this.satelliteGroup.add(body);

        // Solar panels (2 wings)
        const panelGeo = new THREE.BoxGeometry(0.8, 0.12, 0.02);
        const panelMat = new THREE.MeshStandardMaterial({ color: 0x0084FF, emissive: 0x003366, metalness: 0.5 });
        const leftPanel = new THREE.Mesh(panelGeo, panelMat);
        leftPanel.position.x = 0.5;
        const rightPanel = new THREE.Mesh(panelGeo, panelMat);
        rightPanel.position.x = -0.5;
        this.satelliteGroup.add(leftPanel);
        this.satelliteGroup.add(rightPanel);

        // Glowing indicator LED
        const ledGeo = new THREE.SphereGeometry(0.04, 8, 8);
        const ledMat = new THREE.MeshBasicMaterial({ color: 0x00D9FF });
        const led = new THREE.Mesh(ledGeo, ledMat);
        led.position.set(0, 0.1, 0.14);
        this.satelliteGroup.add(led);

        this.scene.add(this.satelliteGroup);

        // Signal Beam Particles
        this.createSignalBeams();
    }

    createSignalBeams() {
        const particleCount = 40;
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount * 3; i++) {
            pos[i] = 0;
        }

        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({
            color: 0x00D9FF,
            size: 0.06,
            transparent: true,
            opacity: 0.85,
            blending: THREE.AdditiveBlending
        });

        this.signalParticles = new THREE.Points(geo, mat);
        this.scene.add(this.signalParticles);
    }

    createStarfield() {
        const starCount = 1500;
        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(starCount * 3);
        const opacities = new Float32Array(starCount);

        for (let i = 0; i < starCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 80;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 80;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 80 - 10;
            opacities[i] = Math.random();
        }

        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('opacity', new THREE.BufferAttribute(opacities, 1));

        const mat = new THREE.PointsMaterial({
            color: 0xFFFFFF,
            size: 0.08,
            transparent: true,
            opacity: 0.8
        });

        this.starField = new THREE.Points(geo, mat);
        this.scene.add(this.starField);
    }

    updateEarthPosition() {
        if (!this.earth) return;
        if (window.innerWidth > 1024) {
            this.earth.position.set(2.8, -0.2, 0);
        } else {
            this.earth.position.set(0, -1.5, 0);
        }
    }

    animate() {
        this.time += 0.016;

        // 1. Slow Earth rotation (1 full turn every ~25s)
        if (this.earth) {
            this.earth.rotation.y += 0.0025;
        }

        // 2. Satellite Elliptical Orbit around Earth
        if (this.satelliteGroup && this.earth) {
            this.orbitAngle += 0.008; // ~45s per orbit

            const rx = 4.2;
            const ry = 2.8;

            const satX = this.earth.position.x + Math.cos(this.orbitAngle) * rx;
            const satY = this.earth.position.y + Math.sin(this.orbitAngle * 0.8) * 0.6;
            const satZ = this.earth.position.z + Math.sin(this.orbitAngle) * ry;

            this.satelliteGroup.position.set(satX, satY, satZ);
            this.satelliteGroup.rotation.y += 0.015;

            // Signal beams firing toward Earth
            if (this.signalParticles) {
                const pos = this.signalParticles.geometry.attributes.position.array;
                for (let i = 0; i < 40; i++) {
                    const frac = (i / 40 + (this.time * 2)) % 1;
                    pos[i * 3] = satX + (this.earth.position.x - satX) * frac;
                    pos[i * 3 + 1] = satY + (this.earth.position.y - satY) * frac;
                    pos[i * 3 + 2] = satZ + (this.earth.position.z - satZ) * frac;
                }
                this.signalParticles.geometry.attributes.position.needsUpdate = true;
            }
        }

        // 3. Twinkle Stars
        if (this.starField) {
            this.starField.rotation.y = this.time * 0.0003;
        }

        this.renderer.render(this.scene, this.camera);
        requestAnimationFrame(() => this.animate());
    }

    resize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.updateEarthPosition();
    }
}

// Auto Instantiate 3D Earth Scene
window.addEventListener('DOMContentLoaded', () => {
    window.earthScene = new EarthScene('webgl-canvas');
});
