/**
 * AETHER-SAR: Space & Radar Slow Motion Background Canvas Engine
 * Renders twinkling space starfield, slow floating nebulae, orbital grid,
 * and a slow-motion 360-degree sweeping polar radar beam with telemetry.
 */

class SpaceRadarBackground {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.stars = [];
        this.blips = [];
        this.radarAngle = 0;
        this.sweepSpeed = 0.003; // Slow motion sweep rate
        this.nebulaNodes = [];
        this.satelliteOrbits = [];
        this.time = 0;

        this.init();
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    init() {
        // Create stars
        this.stars = [];
        const starCount = 350;
        for (let i = 0; i < starCount; i++) {
            this.stars.push({
                x: Math.random(),
                y: Math.random(),
                size: Math.random() * 1.8 + 0.3,
                brightness: Math.random() * 0.8 + 0.2,
                twinkleSpeed: Math.random() * 0.02 + 0.005,
                color: this.getRandomStarColor()
            });
        }

        // Create slow floating nebulae nodes
        this.nebulaNodes = [
            { x: 0.25, y: 0.3, radius: 0.35, color: 'rgba(10, 30, 80, 0.25)', dx: 0.00005, dy: 0.00003 },
            { x: 0.75, y: 0.7, radius: 0.4, color: 'rgba(0, 80, 120, 0.2)', dx: -0.00004, dy: -0.00002 },
            { x: 0.5, y: 0.5, radius: 0.5, color: 'rgba(40, 10, 70, 0.15)', dx: 0.00002, dy: -0.00004 },
            { x: 0.8, y: 0.2, radius: 0.3, color: 'rgba(0, 150, 180, 0.12)', dx: -0.00003, dy: 0.00003 }
        ];

        // Create orbital paths
        this.satelliteOrbits = [
            { radiusX: 0.35, radiusY: 0.18, tilt: -0.25, speed: 0.0005, angle: 0, label: 'SENTINEL-1A [SAR-C]' },
            { radiusX: 0.45, radiusY: 0.22, tilt: 0.35, speed: -0.0004, angle: 1.5, label: 'RADARSAT-2' },
            { radiusX: 0.28, radiusY: 0.14, tilt: 0.6, speed: 0.0007, angle: 3.1, label: 'NISAR L&C-BAND' }
        ];

        // Create persistent radar targets (blips)
        this.generateRadarBlips();
    }

    generateRadarBlips() {
        this.blips = [];
        const count = 12;
        for (let i = 0; i < count; i++) {
            const dist = Math.random() * 0.4 + 0.08; // normalized from radar center
            const angle = Math.random() * Math.PI * 2;
            this.blips.push({
                dist: dist,
                angle: angle,
                size: Math.random() * 3 + 2,
                intensity: 0,
                rcs: (Math.random() * 25 + 5).toFixed(1) + ' dB', // Radar Cross Section
                label: 'TRG-' + Math.floor(1000 + Math.random() * 9000)
            });
        }
    }

    getRandomStarColor() {
        const colors = [
            'rgba(255, 255, 255, ',
            'rgba(180, 220, 255, ',
            'rgba(0, 243, 255, ',
            'rgba(200, 180, 255, '
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    resize() {
        this.width = this.canvas.width = window.innerWidth;
        this.height = this.canvas.height = window.innerHeight;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;
        this.radarRadius = Math.min(this.width, this.height) * 0.42;
    }

    render() {
        this.time += 1;
        this.ctx.clearRect(0, 0, this.width, this.height);

        // 1. Draw Deep Cosmos Background Gradient
        const bgGrad = this.ctx.createRadialGradient(
            this.centerX, this.centerY, 10,
            this.centerX, this.centerY, Math.max(this.width, this.height) * 0.7
        );
        bgGrad.addColorStop(0, '#060a19');
        bgGrad.addColorStop(0.5, '#040612');
        bgGrad.addColorStop(1, '#020308');
        this.ctx.fillStyle = bgGrad;
        this.ctx.fillRect(0, 0, this.width, this.height);

        // 2. Draw Slow Moving Nebulae
        this.drawNebulae();

        // 3. Draw Stars
        this.drawStarfield();

        // 4. Draw Slow Motion Orbital Grid & Trajectories
        this.drawOrbits();

        // 5. Draw Radar Polar Grid
        this.drawRadarGrid();

        // 6. Draw Sweeping Radar Beam & Blips
        this.drawRadarSweep();

        // 7. Draw Telemetry HUD Overlays on Background
        this.drawHUDTelemetry();

        requestAnimationFrame(() => this.render());
    }

    drawNebulae() {
        this.nebulaNodes.forEach(node => {
            node.x += node.dx;
            node.y += node.dy;
            if (node.x < 0.1 || node.x > 0.9) node.dx *= -1;
            if (node.y < 0.1 || node.y > 0.9) node.dy *= -1;

            const cx = node.x * this.width;
            const cy = node.y * this.height;
            const rad = node.radius * Math.min(this.width, this.height);

            const grad = this.ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
            grad.addColorStop(0, node.color);
            grad.addColorStop(1, 'rgba(0,0,0,0)');

            this.ctx.fillStyle = grad;
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, rad, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawStarfield() {
        this.stars.forEach(star => {
            star.brightness += Math.sin(this.time * star.twinkleSpeed) * 0.015;
            const alpha = Math.max(0.1, Math.min(1, star.brightness));

            const sx = star.x * this.width;
            const sy = star.y * this.height;

            this.ctx.fillStyle = star.color + alpha + ')';
            this.ctx.beginPath();
            this.ctx.arc(sx, sy, star.size, 0, Math.PI * 2);
            this.ctx.fill();

            // Subtle glow for larger stars
            if (star.size > 1.4) {
                this.ctx.fillStyle = star.color + (alpha * 0.25) + ')';
                this.ctx.beginPath();
                this.ctx.arc(sx, sy, star.size * 2.5, 0, Math.PI * 2);
                this.ctx.fill();
            }
        });
    }

    drawOrbits() {
        this.satelliteOrbits.forEach(orb => {
            orb.angle += orb.speed;

            this.ctx.save();
            this.ctx.translate(this.centerX, this.centerY);
            this.ctx.rotate(orb.tilt);

            const rx = orb.radiusX * this.width;
            const ry = orb.radiusY * this.height;

            // Draw orbit path line
            this.ctx.strokeStyle = 'rgba(0, 243, 255, 0.08)';
            this.ctx.lineWidth = 1;
            this.ctx.setLineDash([4, 6]);
            this.ctx.beginPath();
            this.ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
            this.ctx.stroke();
            this.ctx.setLineDash([]);

            // Draw satellite position
            const satX = rx * Math.cos(orb.angle);
            const satY = ry * Math.sin(orb.angle);

            // Satellite pulse glow
            const satGrad = this.ctx.createRadialGradient(satX, satY, 0, satX, satY, 12);
            satGrad.addColorStop(0, '#00f3ff');
            satGrad.addColorStop(0.5, 'rgba(0, 243, 255, 0.4)');
            satGrad.addColorStop(1, 'rgba(0, 243, 255, 0)');
            this.ctx.fillStyle = satGrad;
            this.ctx.beginPath();
            this.ctx.arc(satX, satY, 12, 0, Math.PI * 2);
            this.ctx.fill();

            // Satellite solid core
            this.ctx.fillStyle = '#ffffff';
            this.ctx.beginPath();
            this.ctx.arc(satX, satY, 2.5, 0, Math.PI * 2);
            this.ctx.fill();

            // Satellite label
            this.ctx.font = '9px "JetBrains Mono", monospace';
            this.ctx.fillStyle = 'rgba(0, 243, 255, 0.6)';
            this.ctx.fillText(orb.label, satX + 14, satY + 3);

            this.ctx.restore();
        });
    }

    drawRadarGrid() {
        this.ctx.save();
        this.ctx.translate(this.centerX, this.centerY);

        // Concentric distance rings
        const ringRatios = [0.25, 0.5, 0.75, 1.0];
        ringRatios.forEach((ratio, idx) => {
            const r = this.radarRadius * ratio;
            this.ctx.strokeStyle = (idx === 3) ? 'rgba(0, 243, 255, 0.25)' : 'rgba(0, 243, 255, 0.12)';
            this.ctx.lineWidth = (idx === 3) ? 1.5 : 1;
            this.ctx.beginPath();
            this.ctx.arc(0, 0, r, 0, Math.PI * 2);
            this.ctx.stroke();

            // Distance labels
            this.ctx.font = '10px "JetBrains Mono", monospace';
            this.ctx.fillStyle = 'rgba(0, 243, 255, 0.35)';
            this.ctx.fillText(`${(ratio * 200).toFixed(0)} km`, r + 4, -4);
        });

        // Crosshairs & Azimuth lines (30 deg intervals)
        for (let a = 0; a < 360; a += 30) {
            const rad = (a * Math.PI) / 180;
            const x2 = Math.cos(rad) * this.radarRadius;
            const y2 = Math.sin(rad) * this.radarRadius;

            this.ctx.strokeStyle = (a % 90 === 0) ? 'rgba(0, 243, 255, 0.2)' : 'rgba(0, 243, 255, 0.07)';
            this.ctx.lineWidth = (a % 90 === 0) ? 1.2 : 0.8;
            this.ctx.beginPath();
            this.ctx.moveTo(0, 0);
            this.ctx.lineTo(x2, y2);
            this.ctx.stroke();

            // Outer degree markings
            if (a % 30 === 0) {
                const lx = Math.cos(rad) * (this.radarRadius + 14);
                const ly = Math.sin(rad) * (this.radarRadius + 14);
                this.ctx.font = '9px "JetBrains Mono", monospace';
                this.ctx.fillStyle = 'rgba(0, 243, 255, 0.4)';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(`${a}°`, lx, ly);
            }
        }

        this.ctx.restore();
    }

    drawRadarSweep() {
        this.radarAngle += this.sweepSpeed;
        if (this.radarAngle >= Math.PI * 2) {
            this.radarAngle -= Math.PI * 2;
        }

        this.ctx.save();
        this.ctx.translate(this.centerX, this.centerY);

        // 1. Draw Slow Trailing Fade Cone (Sweep gradient)
        const sweepAngleSpan = Math.PI / 3.5; // ~50 degree trail
        const steps = 40;
        for (let i = 0; i < steps; i++) {
            const fraction = i / steps;
            const angleCurrent = this.radarAngle - (sweepAngleSpan * (1 - fraction));
            const alpha = Math.pow(fraction, 2.5) * 0.25; // exponential trail glow

            this.ctx.fillStyle = `rgba(0, 243, 255, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.moveTo(0, 0);
            this.ctx.arc(0, 0, this.radarRadius, angleCurrent, angleCurrent + (sweepAngleSpan / steps) + 0.005);
            this.ctx.closePath();
            this.ctx.fill();
        }

        // 2. Leading Edge Beam Line
        const leadX = Math.cos(this.radarAngle) * this.radarRadius;
        const leadY = Math.sin(this.radarAngle) * this.radarRadius;

        const beamGrad = this.ctx.createLinearGradient(0, 0, leadX, leadY);
        beamGrad.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
        beamGrad.addColorStop(0.5, 'rgba(0, 243, 255, 0.8)');
        beamGrad.addColorStop(1, 'rgba(0, 243, 255, 0)');

        this.ctx.strokeStyle = beamGrad;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(0, 0);
        this.ctx.lineTo(leadX, leadY);
        this.ctx.stroke();

        // 3. Draw & Illuminate Radar Targets (Blips)
        this.blips.forEach(blip => {
            const blipAngle = blip.angle;
            // Calculate angular separation from sweep beam
            let diff = this.radarAngle - blipAngle;
            while (diff < 0) diff += Math.PI * 2;
            while (diff >= Math.PI * 2) diff -= Math.PI * 2;

            // Illuminate if beam just passed over (diff between 0 and 0.5 rad)
            if (diff < 0.6) {
                blip.intensity = 1.0 - (diff / 0.6);
            } else {
                blip.intensity *= 0.985; // slow decay
            }

            if (blip.intensity > 0.05) {
                const bx = Math.cos(blipAngle) * (blip.dist * this.radarRadius * 2);
                const by = Math.sin(blipAngle) * (blip.dist * this.radarRadius * 2);

                // Blip pulse ring
                this.ctx.strokeStyle = `rgba(0, 255, 136, ${blip.intensity * 0.7})`;
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.arc(bx, by, blip.size * 2 * (2 - blip.intensity), 0, Math.PI * 2);
                this.ctx.stroke();

                // Blip dot
                this.ctx.fillStyle = `rgba(0, 255, 136, ${blip.intensity})`;
                this.ctx.beginPath();
                this.ctx.arc(bx, by, blip.size, 0, Math.PI * 2);
                this.ctx.fill();

                // Target tag text
                if (blip.intensity > 0.4) {
                    this.ctx.font = '9px "JetBrains Mono", monospace';
                    this.ctx.fillStyle = `rgba(0, 255, 136, ${blip.intensity * 0.8})`;
                    this.ctx.fillText(`${blip.label} [${blip.rcs}]`, bx + 8, by - 4);
                }
            }
        });

        this.ctx.restore();
    }

    drawHUDTelemetry() {
        this.ctx.save();

        // Top Left Telemetry Readout
        this.ctx.font = '10px "JetBrains Mono", monospace';
        this.ctx.fillStyle = 'rgba(0, 243, 255, 0.45)';

        const deg = ((this.radarAngle * 180) / Math.PI).toFixed(1);
        this.ctx.fillText(`AZIMUTH SWEEP: ${deg.padStart(5, '0')}° [SLOW-MO]`, 25, this.height - 50);
        this.ctx.fillText(`SENSOR MODE: STRIPMAP / C-BAND (5.405 GHz)`, 25, this.height - 35);
        this.ctx.fillText(`POLARIZATION: VV + VH DUAL-POL`, 25, this.height - 20);

        // Top Right Corner Coordinates
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`ORBITAL SPEED: 7.56 KM/S`, this.width - 25, this.height - 35);
        this.ctx.fillText(`SPACEBORNE RADAR SYSTEM v3.4`, this.width - 25, this.height - 20);

        this.ctx.restore();
    }
}

// Instantiate on load
window.addEventListener('DOMContentLoaded', () => {
    window.spaceRadarBg = new SpaceRadarBackground('space-radar-canvas');
    window.spaceRadarBg.render();
});
