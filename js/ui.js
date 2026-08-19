/**
 * AETHER 3D SAR COLORIZER - UI CONTROLLER & INTERACTION FLOW
 * Manages 5-step user workflow (Upload -> Process -> Results -> Download/Reset)
 * along with preset selection, before/after slider, and FAQ modal.
 */

class SARUIController {
    constructor() {
        // Cards
        this.cardUpload = document.getElementById('card-upload');
        this.cardProcessing = document.getElementById('card-processing');
        this.cardResults = document.getElementById('card-results');

        // Dropzone & File Input
        this.dropzone = document.getElementById('dropzone');
        this.fileInput = document.getElementById('file-input');

        // Progress & Status
        this.progressFill = document.getElementById('progress-fill');
        this.progressPercent = document.getElementById('progress-percent');
        this.statusMsg = document.getElementById('status-msg');

        // Comparison Canvas Stage
        this.compCanvasBase = document.getElementById('comp-canvas-base');
        this.compCanvasClip = document.getElementById('comp-canvas-clip');
        this.compClipWrapper = document.getElementById('comp-clip-wrapper');
        this.compSliderInput = document.getElementById('comp-slider-input');

        // Buttons
        this.btnDownload = document.getElementById('btn-download');
        this.btnReset = document.getElementById('btn-reset');
        this.faqFab = document.getElementById('faq-fab');
        this.faqModal = document.getElementById('faq-modal');
        this.modalClose = document.getElementById('modal-close');

        // Current Preset State
        this.currentPreset = 'coastal';
        this.isProcessing = false;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupComparisonCanvases();
    }

    setupEventListeners() {
        // Drag & Drop
        if (this.dropzone) {
            this.dropzone.addEventListener('click', () => this.fileInput.click());
            this.dropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                this.dropzone.classList.add('drag-over');
            });
            this.dropzone.addEventListener('dragleave', () => {
                this.dropzone.classList.remove('drag-over');
            });
            this.dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                this.dropzone.classList.remove('drag-over');
                if (e.dataTransfer.files.length > 0) {
                    this.startProcessingFlow(e.dataTransfer.files[0].name);
                }
            });
        }

        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.startProcessingFlow(e.target.files[0].name);
                }
            });
        }

        // Preset Buttons
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentPreset = btn.getAttribute('data-preset');
                this.startProcessingFlow(`Preset: ${this.currentPreset.toUpperCase()}`);
            });
        });

        // Comparison Slider
        if (this.compSliderInput) {
            this.compSliderInput.addEventListener('input', (e) => {
                const val = e.target.value;
                if (this.compClipWrapper) {
                    this.compClipWrapper.style.width = `${val}%`;
                }
            });
        }

        // Action Buttons
        if (this.btnReset) {
            this.btnReset.addEventListener('click', () => this.resetFlow());
        }

        if (this.btnDownload) {
            this.btnDownload.addEventListener('click', () => this.downloadResult());
        }

        // FAQ Modal
        if (this.faqFab && this.faqModal) {
            this.faqFab.addEventListener('click', () => this.faqModal.classList.add('active'));
        }
        if (this.modalClose && this.faqModal) {
            this.modalClose.addEventListener('click', () => this.faqModal.classList.remove('active'));
        }
        if (this.faqModal) {
            this.faqModal.addEventListener('click', (e) => {
                if (e.target === this.faqModal) this.faqModal.classList.remove('active');
            });
        }
    }

    startProcessingFlow(fileName) {
        if (this.isProcessing) return;
        this.isProcessing = true;

        // Hide Card 1, Show Card 2
        this.cardUpload.classList.add('hidden');
        this.cardProcessing.classList.remove('hidden');
        this.cardResults.classList.add('hidden');

        // Animate Progress Bar & Messages
        let progress = 0;
        const messages = [
            'Analyzing SAR Radar Data...',
            'Filtering Speckle Noise...',
            'Extracting Dielectric Features...',
            'Predicting LAB Chrominance...',
            'Finalizing GeoTIFF Colorization...'
        ];

        const interval = setInterval(() => {
            progress += 2;
            if (this.progressFill) this.progressFill.style.width = `${progress}%`;
            if (this.progressPercent) this.progressPercent.innerText = `${progress}%`;

            const msgIdx = Math.min(messages.length - 1, Math.floor((progress / 100) * messages.length));
            if (this.statusMsg) this.statusMsg.innerText = messages[msgIdx];

            if (progress >= 100) {
                clearInterval(interval);
                setTimeout(() => this.showResults(), 400);
            }
        }, 120); // ~6 seconds total simulation
    }

    showResults() {
        this.isProcessing = false;

        // Hide Card 2, Show Card 3
        this.cardProcessing.classList.add('hidden');
        this.cardResults.classList.remove('hidden');

        // Draw Images to Canvases based on preset
        this.drawComparisonData(this.currentPreset);
    }

    resetFlow() {
        this.isProcessing = false;
        if (this.progressFill) this.progressFill.style.width = '0%';
        if (this.progressPercent) this.progressPercent.innerText = '0%';

        this.cardResults.classList.add('hidden');
        this.cardProcessing.classList.add('hidden');
        this.cardUpload.classList.remove('hidden');
    }

    setupComparisonCanvases() {
        if (!this.compCanvasBase || !this.compCanvasClip) return;
        this.compCanvasBase.width = this.compCanvasClip.width = 600;
        this.compCanvasBase.height = this.compCanvasClip.height = 360;
        this.drawComparisonData('coastal');
    }

    drawComparisonData(presetKey) {
        const ctxBase = this.compCanvasBase.getContext('2d');
        const ctxClip = this.compCanvasClip.getContext('2d');
        const w = 600, h = 360;

        const imgBase = ctxBase.createImageData(w, h);
        const imgClip = ctxClip.createImageData(w, h);

        const dBase = imgBase.data;
        const dClip = imgClip.data;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const nx = x / w;
                const ny = y / h;

                let val = Math.sin(nx * 10) * Math.cos(ny * 8) * 128 + 128;
                let noise = (Math.random() - 0.5) * 40;

                let sarVal = Math.max(0, Math.min(255, val + noise));

                // B&W SAR
                dBase[idx] = sarVal;
                dBase[idx + 1] = sarVal;
                dBase[idx + 2] = sarVal;
                dBase[idx + 3] = 255;

                // Colorized Optical
                let r = 40, g = 180, b = 220;
                if (presetKey === 'metropolis') { r = 210; g = 160; b = 150; }
                else if (presetKey === 'delta') { r = 30; g = 190; b = 60; }
                else if (presetKey === 'agriculture') { r = 220; g = 170; b = 50; }

                dClip[idx] = Math.max(0, Math.min(255, r + noise * 0.5));
                dClip[idx + 1] = Math.max(0, Math.min(255, g + noise * 0.5));
                dClip[idx + 2] = Math.max(0, Math.min(255, b + noise * 0.5));
                dClip[idx + 3] = 255;
            }
        }

        ctxBase.putImageData(imgBase, 0, 0);
        ctxClip.putImageData(imgClip, 0, 0);
    }

    downloadResult() {
        const link = document.createElement('a');
        link.download = `SAR_COLORIZED_${this.currentPreset.toUpperCase()}_HD.png`;
        link.href = this.compCanvasClip.toDataURL('image/png');
        link.click();
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.sarUI = new SARUIController();
});
