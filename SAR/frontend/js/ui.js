/**
 * AETHER 3D SAR COLORIZER - UI CONTROLLER & INTERACTION FLOW
 * Manages 5-step user workflow (Upload -> Process -> Results -> Download/Reset)
 * along with preset selection, before/after slider, and FAQ modal.
 * Supports dual modes: Realistic Colorization and Classified Land-Cover Map.
 * Reuses identical curtain slider component for both Colorize & Classify modes.
 * Supports confidence heatmap toggle in classification mode.
 * Supports optional Ground-Truth optical image evaluation (PSNR & SSIM metrics).
 * Generates client-side high-resolution Analysis Report PNGs.
 * Manages collapsible History Panel connected to backend SQLite storage.
 */

class SARUIController {
    constructor() {
        // Dropzone & File Input (Primary SAR)
        this.dropzone = document.getElementById('dropzone');
        this.fileInput = document.getElementById('file-input');

        // Dropzone & File Input (Ground-Truth Optical)
        this.gtDropzone = document.getElementById('gt-dropzone');
        this.gtFileInput = document.getElementById('gt-file-input');
        this.gtFileName = document.getElementById('gt-file-name');

        // Metrics Banner Elements
        this.metricsBanner = document.getElementById('metrics-banner');
        this.metricPsnr = document.getElementById('metric-psnr');
        this.metricSsim = document.getElementById('metric-ssim');

        // Buttons
        this.btnRunModel = document.getElementById('btn-run-model');
        this.btnRunLabel = document.getElementById('btn-run-label');
        this.btnDownload = document.getElementById('btn-download');
        this.btnDownloadReport = document.getElementById('btn-download-report');
        this.btnCompareBoth = document.getElementById('btn-compare-both');
        this.faqFab = document.getElementById('faq-fab');
        this.faqModal = document.getElementById('faq-modal');
        this.modalClose = document.getElementById('modal-close');

        // Comparison Canvas Stage (Main)
        this.compCanvasBase = document.getElementById('comp-canvas-base');
        this.compCanvasClip = document.getElementById('comp-canvas-clip');
        this.compClipWrapper = document.getElementById('comp-clip-wrapper');
        this.compSliderInput = document.getElementById('comp-slider-input');

        // Mode Toggle & Legend Elements
        this.modeToggle = document.getElementById('mode-toggle');
        this.badgeLeft = document.getElementById('badge-left');
        this.legend = document.getElementById('land-cover-legend');
        this.confidenceCheckbox = document.getElementById('confidence-checkbox');
        this.confidenceScaleRow = document.getElementById('confidence-scale-row');
        this.classSwatchesRow = document.getElementById('class-swatches-row');

        // Compare Triple Panel Elements (Dual Sliders)
        this.compareTriple = document.getElementById('compare-triple');
        this.compareSliderColorize = document.getElementById('compare-slider-colorize');
        this.compareClipColorize = document.getElementById('compare-clip-colorize');
        this.compareSliderClassify = document.getElementById('compare-slider-classify');
        this.compareClipClassify = document.getElementById('compare-clip-classify');

        // Mobile Drawer Elements
        this.workspaceSidebar = document.querySelector('.workspace-sidebar');
        this.mobileDrawerToggle = document.getElementById('mobile-drawer-toggle');
        this.sidebarBackdrop = document.getElementById('sidebar-backdrop');

        // History Panel Elements
        this.historyHeader = document.getElementById('history-header');
        this.historyContent = document.getElementById('history-content');
        this.historyList = document.getElementById('history-list');
        this.historyChevron = document.getElementById('history-chevron');

        // Change Detection Elements
        this.singleUploadSection = document.getElementById('single-upload-section');
        this.changeDetectControls = document.getElementById('change-detect-controls');
        this.dropzoneBefore = document.getElementById('dropzone-before');
        this.fileInputBefore = document.getElementById('file-input-before');
        this.beforeFileName = document.getElementById('before-file-name');
        this.dropzoneAfter = document.getElementById('dropzone-after');
        this.fileInputAfter = document.getElementById('file-input-after');
        this.afterFileName = document.getElementById('after-file-name');
        this.pixelResolutionInput = document.getElementById('pixel-resolution-input');

        // Fusion Elements
        this.fusionControls = document.getElementById('fusion-controls');
        this.dropzoneFusionSar = document.getElementById('dropzone-fusion-sar');
        this.fileInputFusionSar = document.getElementById('file-input-fusion-sar');
        this.fusionSarFileName = document.getElementById('fusion-sar-file-name');
        this.dropzoneFusionOpt = document.getElementById('dropzone-fusion-opt');
        this.fileInputFusionOpt = document.getElementById('file-input-fusion-opt');
        this.fusionOptFileName = document.getElementById('fusion-opt-file-name');

        this.lastFusionSarFile = null;
        this.lastFusionOptFile = null;

        this.changeStageWrapper = document.getElementById('change-stage-wrapper');
        this.changeHeadlineText = document.getElementById('change-headline-text');
        this.changeCanvasBefore = document.getElementById('change-canvas-before');
        this.changeCanvasAfter = document.getElementById('change-canvas-after');
        this.changeCanvasOverlay = document.getElementById('change-canvas-overlay');
        this.changeTableBody = document.getElementById('change-table-body');
        this.comparisonContainer = document.getElementById('comparison-container');

        this.lastBeforeFile = null;
        this.lastAfterFile = null;

        // Narrative Elements
        this.btnNarrateClassify = document.getElementById('btn-generate-narrative-classify');
        this.narrativeBoxClassify = document.getElementById('narrative-box-classify');
        this.narrativeTextClassify = document.getElementById('narrative-text-classify');

        this.btnNarrateChange = document.getElementById('btn-generate-narrative-change');
        this.narrativeBoxChange = document.getElementById('narrative-box-change');
        this.narrativeTextChange = document.getElementById('narrative-text-change');

        this.lastClassifyStats = null;
        this.lastChangeStats = null;

        // Current State
        this.currentPreset = 'coastal';
        this.currentMode = 'colorize'; // 'colorize', 'classify', or 'change_detect'
        this.isProcessing = false;
        this.currentFetchPromise = null;
        this.lastUploadedFile = null; // Keep reference for Compare Both
        this.lastGroundTruthFile = null; // Ground-Truth reference file

        // Cache for classify results & evaluation metrics
        this.classifiedImageSrc = null;
        this.confidenceHeatmapSrc = null;
        this.lastInferenceTimeMs = null;
        this.lastPsnr = null;
        this.lastSsim = null;
        this.lastModelCheckpoint = 'gan_final_epoch24';

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupComparisonCanvases();
        this.fetchHistory();
    }

    setupEventListeners() {
        // --- Mobile Drawer Toggle ---
        if (this.mobileDrawerToggle && this.workspaceSidebar) {
            this.mobileDrawerToggle.addEventListener('click', () => {
                const isOpen = this.workspaceSidebar.classList.toggle('drawer-open');
                if (this.sidebarBackdrop) {
                    this.sidebarBackdrop.classList.toggle('active', isOpen);
                }
            });
        }

        if (this.sidebarBackdrop && this.workspaceSidebar) {
            this.sidebarBackdrop.addEventListener('click', () => {
                if (this.workspaceSidebar) this.workspaceSidebar.classList.remove('drawer-open');
                if (this.sidebarBackdrop) this.sidebarBackdrop.classList.remove('active');
            });
        }

        // --- Parameterized Curtain Sliders Binding (Main Stage + Compare Panels) ---
        bindCurtainSlider(this.compSliderInput, this.compClipWrapper);
        bindCurtainSlider(this.compareSliderColorize, this.compareClipColorize);
        bindCurtainSlider(this.compareSliderClassify, this.compareClipClassify);

        // --- History Panel Collapsible Toggle ---
        if (this.historyHeader && this.historyContent) {
            this.historyHeader.addEventListener('click', () => {
                const isCollapsed = this.historyContent.classList.toggle('collapsed');
                if (this.historyChevron) {
                    this.historyChevron.style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)';
                }
            });
        }

        // --- Mode Toggle ---
        if (this.modeToggle) {
            this.modeToggle.querySelectorAll('.mode-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.modeToggle.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.currentMode = btn.getAttribute('data-mode');
                    this.updateModeUI();
                });
            });
        }

        // --- Confidence Heatmap Checkbox ---
        if (this.confidenceCheckbox) {
            this.confidenceCheckbox.addEventListener('change', () => {
                this.toggleConfidenceDisplay();
            });
        }

        // --- Primary SAR Drag & Drop ---
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
                    const file = e.dataTransfer.files[0];
                    this.lastUploadedFile = file;
                    this.hideCompareTriple();
                    this.currentFetchPromise = processSARFile(file, this.currentMode, this.lastGroundTruthFile);
                    this.startProcessingFlow(file.name);
                }
            });
        }

        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    const file = e.target.files[0];
                    this.lastUploadedFile = file;
                    this.hideCompareTriple();
                    this.currentFetchPromise = processSARFile(file, this.currentMode, this.lastGroundTruthFile);
                    this.startProcessingFlow(file.name);
                }
            });
        }

        // --- Optional Ground-Truth Optical Upload ---
        if (this.gtDropzone && this.gtFileInput) {
            this.gtDropzone.addEventListener('click', () => this.gtFileInput.click());
            this.gtDropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                this.gtDropzone.classList.add('drag-over');
            });
            this.gtDropzone.addEventListener('dragleave', () => {
                this.gtDropzone.classList.remove('drag-over');
            });
            this.gtDropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                this.gtDropzone.classList.remove('drag-over');
                if (e.dataTransfer.files.length > 0) {
                    this.setGroundTruthFile(e.dataTransfer.files[0]);
                }
            });

            this.gtFileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.setGroundTruthFile(e.target.files[0]);
                }
            });
        }

        // --- Preset Buttons ---
        document.querySelectorAll('.btn-preset, .p-card').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                document.querySelectorAll('.btn-preset, .p-card').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const presetKey = btn.getAttribute('data-preset');
                this.currentPreset = presetKey;

                // Map preset keys to actual filenames in frontend/sample-images/
                const filenameMap = {
                    'coastal': 'coastal.png',
                    'metropolis': 'metropolis.png',
                    'delta': 'river-delta.png',
                    'agriculture': 'crop-plots.png'
                };

                const filename = filenameMap[presetKey] || `${presetKey}.png`;
                const samplePath = `sample-images/${filename}`;

                try {
                    // Fetch local sample image and convert to a File object
                    const res = await fetch(samplePath);
                    if (!res.ok) throw new Error(`Could not load sample image at ${samplePath}`);
                    const blob = await res.blob();
                    const file = new File([blob], filename, { type: blob.type || 'image/png' });

                    this.lastUploadedFile = file;
                    this.hideCompareTriple();

                    // Trigger AI pipeline via backend using current mode
                    this.currentFetchPromise = processSARFile(file, this.currentMode, this.lastGroundTruthFile);
                    this.startProcessingFlow(`Preset: ${presetKey.toUpperCase()}`);
                } catch (err) {
                    console.error('Failed to load preset sample image:', err);
                    alert(`Could not load preset sample image (${samplePath}). Ensure the file exists in frontend/sample-images/.`);
                }
            });
        });

        // --- Change Detection Dropzones ---
        if (this.dropzoneBefore && this.fileInputBefore) {
            this.dropzoneBefore.addEventListener('click', () => this.fileInputBefore.click());
            this.fileInputBefore.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.lastBeforeFile = e.target.files[0];
                    if (this.beforeFileName) this.beforeFileName.textContent = `Before: ${this.lastBeforeFile.name}`;
                    this.dropzoneBefore.classList.add('active-gt');
                }
            });
        }

        if (this.dropzoneAfter && this.fileInputAfter) {
            this.dropzoneAfter.addEventListener('click', () => this.fileInputAfter.click());
            this.fileInputAfter.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.lastAfterFile = e.target.files[0];
                    if (this.afterFileName) this.afterFileName.textContent = `After: ${this.lastAfterFile.name}`;
                    this.dropzoneAfter.classList.add('active-gt');
                }
            });
        }

        // --- Fusion Dropzones ---
        if (this.dropzoneFusionSar && this.fileInputFusionSar) {
            this.dropzoneFusionSar.addEventListener('click', () => this.fileInputFusionSar.click());
            this.fileInputFusionSar.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.lastFusionSarFile = e.target.files[0];
                    if (this.fusionSarFileName) this.fusionSarFileName.textContent = `SAR: ${this.lastFusionSarFile.name}`;
                    this.dropzoneFusionSar.classList.add('active-gt');
                }
            });
        }

        if (this.dropzoneFusionOpt && this.fileInputFusionOpt) {
            this.dropzoneFusionOpt.addEventListener('click', () => this.fileInputFusionOpt.click());
            this.fileInputFusionOpt.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.lastFusionOptFile = e.target.files[0];
                    if (this.fusionOptFileName) this.fusionOptFileName.textContent = `Optical: ${this.lastFusionOptFile.name}`;
                    this.dropzoneFusionOpt.classList.add('active-gt');
                }
            });
        }

        // --- Action Buttons ---
        if (this.btnRunModel) {
            this.btnRunModel.addEventListener('click', () => {
                if (this.currentMode === 'fusion') {
                    this.processFusion();
                } else if (this.currentMode === 'change_detect') {
                    this.processChangeDetection();
                } else if (this.lastUploadedFile) {
                    this.currentFetchPromise = processSARFile(this.lastUploadedFile, this.currentMode, this.lastGroundTruthFile);
                    this.startProcessingFlow(this.lastUploadedFile.name);
                } else {
                    alert('Please upload a SAR image first.');
                }
            });
        }

        if (this.btnReset) {
            this.btnReset.addEventListener('click', () => this.resetFlow());
        }

        if (this.btnDownload) {
            this.btnDownload.addEventListener('click', () => this.downloadResult());
        }

        if (this.btnDownloadReport) {
            this.btnDownloadReport.addEventListener('click', () => this.generateAnalysisReport());
        }

        // --- Narrative Generation Buttons ---
        if (this.btnNarrateClassify) {
            this.btnNarrateClassify.addEventListener('click', () => this.generateNarrative('classify'));
        }
        if (this.btnNarrateChange) {
            this.btnNarrateChange.addEventListener('click', () => this.generateNarrative('change-detect'));
        }

        // --- Compare Both ---
        if (this.btnCompareBoth) {
            this.btnCompareBoth.addEventListener('click', () => this.runCompareBoth());
        }

        // --- FAQ Modal ---
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

    setGroundTruthFile(file) {
        this.lastGroundTruthFile = file;
        if (this.gtFileName) {
            this.gtFileName.textContent = `GT Attached: ${file.name}`;
        }
        if (this.gtDropzone) {
            this.gtDropzone.classList.add('active-gt');
        }
    }

    // --- Mode UI Updates ---
    updateModeUI() {
        const isClassify = this.currentMode === 'classify';
        const isChangeDetect = this.currentMode === 'change_detect';
        const isFusion = this.currentMode === 'fusion';

        // Single upload vs Change Detect vs Fusion controls
        if (this.singleUploadSection) {
            this.singleUploadSection.classList.toggle('hidden', isChangeDetect || isFusion);
        }
        if (this.changeDetectControls) {
            this.changeDetectControls.classList.toggle('hidden', !isChangeDetect);
        }
        if (this.fusionControls) {
            this.fusionControls.classList.toggle('hidden', !isFusion);
        }

        // Compare Both button (hidden in Change Detection & Fusion)
        if (this.btnCompareBoth) {
            this.btnCompareBoth.classList.toggle('hidden', isChangeDetect || isFusion);
        }

        // Stage containers visibility
        if (this.changeStageWrapper) {
            this.changeStageWrapper.classList.add('hidden');
        }
        if (this.comparisonContainer && !isChangeDetect) {
            this.comparisonContainer.classList.remove('hidden');
        }

        // Update RUN button label
        if (this.btnRunLabel) {
            if (isFusion) {
                this.btnRunLabel.textContent = 'RUN FUSION MODEL';
            } else if (isChangeDetect) {
                this.btnRunLabel.textContent = 'RUN CHANGE DETECTION';
            } else if (isClassify) {
                this.btnRunLabel.textContent = 'RUN CLASSIFICATION MODEL';
            } else {
                this.btnRunLabel.textContent = 'RUN COLORIZATION MODEL';
            }
        }

        // Update left badge default text
        if (this.badgeLeft) {
            if (isFusion) {
                this.badgeLeft.textContent = 'FUSED OPTICAL SPECTRUM [SAR+OPT]';
            } else if (isClassify) {
                this.badgeLeft.textContent = 'CLASSIFIED LAND-COVER MAP';
            } else {
                this.badgeLeft.textContent = 'AI PREDICTED OPTICAL [LAB-TO-RGB]';
            }
        }

        // Show/hide legend + stats panel
        if (this.legend) {
            this.legend.classList.toggle('hidden', !isClassify);
        }
        const statsPanel = document.getElementById('class-stats-panel');
        if (statsPanel && !isClassify) {
            statsPanel.classList.add('hidden');
        }

        // Reset confidence checkbox when changing modes
        if (this.confidenceCheckbox) {
            this.confidenceCheckbox.checked = false;
        }
        if (this.confidenceScaleRow) {
            this.confidenceScaleRow.classList.add('hidden');
        }
        if (this.classSwatchesRow) {
            this.classSwatchesRow.classList.remove('hidden');
        }

        // Hide metrics banner in classify or change_detect mode
        if ((isClassify || isChangeDetect) && this.metricsBanner) {
            this.metricsBanner.classList.add('hidden');
        }
    }

    // --- Toggle Confidence Display ---
    async toggleConfidenceDisplay() {
        if (this.currentMode !== 'classify') return;

        const showConfidence = this.confidenceCheckbox && this.confidenceCheckbox.checked;

        if (this.confidenceScaleRow) {
            this.confidenceScaleRow.classList.toggle('hidden', !showConfidence);
        }
        if (this.classSwatchesRow) {
            this.classSwatchesRow.classList.toggle('hidden', showConfidence);
        }

        const targetSrc = showConfidence ? this.confidenceHeatmapSrc : this.classifiedImageSrc;

        if (targetSrc && this.compCanvasClip) {
            const container = document.getElementById('comparison-container');
            const cw = container?.clientWidth || 600;
            const ch = cw;

            await drawImageToCanvas(targetSrc, this.compCanvasClip, cw, ch);

            if (this.badgeLeft) {
                const ms = parseFloat(this.lastInferenceTimeMs || '0');
                const compact = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
                if (showConfidence) {
                    this.badgeLeft.innerText = `CONFIDENCE HEATMAP [${compact}]`;
                } else {
                    this.badgeLeft.innerText = `CLASSIFIED LAND-COVER [${compact}]`;
                }
            }
        }
    }

    // --- Fetch & Render History ---
    async fetchHistory() {
        if (!this.historyList) return;

        try {
            const response = await fetch('http://localhost:8000/history');
            if (!response.ok) return;

            const records = await response.json();

            if (!Array.isArray(records) || records.length === 0) {
                this.historyList.innerHTML = '<div class="history-empty">No inference records yet</div>';
                return;
            }

            this.historyList.innerHTML = '';

            records.forEach(record => {
                const card = document.createElement('div');
                card.className = 'history-card';

                const imgUrl = `http://localhost:8000/history/${record.id}/image`;
                const relTime = getRelativeTimeString(record.timestamp);

                let tagClass = 'tag-colorize';
                let modeLabel = 'COLORIZE';
                if (record.mode === 'classify') {
                    tagClass = 'tag-classify';
                    modeLabel = 'CLASSIFY';
                } else if (record.mode === 'change_detect' || record.mode === 'change-detect') {
                    tagClass = 'tag-change';
                    modeLabel = 'CHANGE DETECT';
                }

                card.innerHTML = `
                    <img class="history-thumb" src="${imgUrl}" alt="${record.filename}" />
                    <div class="history-info">
                        <span class="history-mode-tag ${tagClass}">${modeLabel}</span>
                        <span class="history-fn">${record.filename}</span>
                        <span class="history-time">${relTime} • ${record.inference_time_ms}ms</span>
                    </div>
                    <button class="history-delete-btn" title="Delete Record">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                    </button>
                `;

                // Click card -> Load result into comparison stage (read-only)
                card.addEventListener('click', (e) => {
                    if (e.target.closest('.history-delete-btn')) return; // Ignore delete click
                    this.loadHistoryItemIntoStage(record, imgUrl);
                });

                // Click delete button -> Delete record from backend & remove card
                const deleteBtn = card.querySelector('.history-delete-btn');
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    try {
                        const delRes = await fetch(`http://localhost:8000/history/${record.id}`, { method: 'DELETE' });
                        if (delRes.ok) {
                            card.remove();
                            if (this.historyList.children.length === 0) {
                                this.historyList.innerHTML = '<div class="history-empty">No inference records yet</div>';
                            }
                        }
                    } catch (err) {
                        console.error('Failed to delete history record:', err);
                    }
                });

                this.historyList.appendChild(card);
            });
        } catch (err) {
            console.warn('Could not load history:', err);
        }
    }

    // --- Load History Item into Comparison Stage (Read-only view) ---
    async loadHistoryItemIntoStage(record, imgUrl) {
        if (record.mode === 'change_detect' || record.mode === 'change-detect') {
            if (this.comparisonContainer) this.comparisonContainer.classList.add('hidden');
            if (this.compareTriple) this.compareTriple.classList.add('hidden');
            if (this.legend) this.legend.classList.add('hidden');
            if (this.changeStageWrapper) this.changeStageWrapper.classList.remove('hidden');

            const panelW = 350;
            const panelH = 350;
            if (this.changeCanvasOverlay) await drawImageToCanvas(imgUrl, this.changeCanvasOverlay, panelW, panelH);
            if (this.changeHeadlineText) {
                const relTime = getRelativeTimeString(record.timestamp);
                this.changeHeadlineText.textContent = `HISTORIC CHANGE OVERLAY MAP [${record.filename} • ${relTime}]`;
            }
            return;
        }

        if (this.changeStageWrapper) this.changeStageWrapper.classList.add('hidden');
        if (this.comparisonContainer) this.comparisonContainer.classList.remove('hidden');
        if (!this.compCanvasClip) return;

        const container = document.getElementById('comparison-container');
        const cw = container?.clientWidth || 600;
        const ch = container?.clientHeight || 520;

        await drawImageToCanvas(imgUrl, this.compCanvasClip, cw, ch);

        if (this.badgeLeft) {
            const relTime = getRelativeTimeString(record.timestamp);
            const modeName = record.mode === 'classify' ? 'HISTORIC CLASSIFICATION' : 'HISTORIC COLORIZATION';
            this.badgeLeft.innerText = `${modeName} [${relTime}]`;
        }

        // Display PSNR & SSIM metrics if recorded
        if (record.psnr && record.ssim && this.metricsBanner) {
            if (this.metricPsnr) this.metricPsnr.textContent = record.psnr;
            if (this.metricSsim) this.metricSsim.textContent = record.ssim;
            this.metricsBanner.classList.remove('hidden');
        } else if (this.metricsBanner) {
            this.metricsBanner.classList.add('hidden');
        }
    }

    // --- Processing Flow ---
    startProcessingFlow(fileName) {
        if (this.isProcessing) return;
        this.isProcessing = true;

        // Disable controls & show loading state on RUN MODEL button
        if (this.btnRunModel) {
            this.btnRunModel.disabled = true;
            this.btnRunModel.dataset.origText = this.btnRunModel.innerHTML;
            this.btnRunModel.innerHTML = `
                <svg class="spin" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
                PROCESSING MODEL...
            `;
        }
        if (this.dropzone) {
            this.dropzone.style.pointerEvents = 'none';
            this.dropzone.style.opacity = '0.6';
        }

        setTimeout(() => this.showResults(), 400);
    }

    async showResults() {
        // If a real file colorization/classification was triggered, await the result and draw real images
        if (this.currentFetchPromise) {
            try {
                const result = await this.currentFetchPromise;

                this.lastInferenceTimeMs = result.inferenceTimeMs;
                this.lastPsnr = result.psnr;
                this.lastSsim = result.ssim;

                // Persist latest metrics to localStorage for cross-page use (benchmarks.html)
                if (result.psnr) localStorage.setItem('aether_last_psnr', result.psnr);
                if (result.ssim) localStorage.setItem('aether_last_ssim', result.ssim);
                if (result.inferenceTimeMs) localStorage.setItem('aether_last_inference_ms', result.inferenceTimeMs);
                this.lastModelCheckpoint = result.modelCheckpoint || (result.mode === 'classify' ? 'classifier' : 'gan_final_epoch24');

                // Refresh history list automatically after every successful run
                this.fetchHistory();

                // Handle evaluation metrics display (PSNR & SSIM)
                if (result.psnr && result.ssim && this.metricsBanner) {
                    if (this.metricPsnr) this.metricPsnr.textContent = result.psnr;
                    if (this.metricSsim) this.metricSsim.textContent = result.ssim;
                    this.metricsBanner.classList.remove('hidden');
                } else if (this.metricsBanner) {
                    this.metricsBanner.classList.add('hidden');
                }

                if (result.mode === 'classify' && result.confidenceHeatmap) {
                    this.classifiedImageSrc = result.classifiedImage;
                    this.confidenceHeatmapSrc = result.confidenceHeatmap;

                    // Cache stats for narrative generation
                    if (result.classPercentages) {
                        this.lastClassifyStats = {
                            class_percentages: result.classPercentages,
                            mean_confidence: result.meanConfidence ?? 0,
                        };
                        this.renderClassStats(result.classPercentages, result.meanConfidence);
                    }

                    // Respect current confidence checkbox toggle state
                    const showConfidence = this.confidenceCheckbox && this.confidenceCheckbox.checked;
                    const activeSrc = showConfidence ? this.confidenceHeatmapSrc : this.classifiedImageSrc;

                    if (this.compCanvasBase && this.compCanvasClip) {
                        const container = document.getElementById('comparison-container');
                        const cw = container?.clientWidth || 700;
                        const ch = cw;

                        await drawImageToCanvas(result.originalFile, this.compCanvasBase, cw, ch);
                        await drawImageToCanvas(activeSrc, this.compCanvasClip, cw, ch);
                    }

                    if (this.badgeLeft && result.inferenceTimeMs) {
                        const ms = parseFloat(result.inferenceTimeMs);
                        const compact = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
                        if (showConfidence) {
                            this.badgeLeft.innerText = `CONFIDENCE HEATMAP [${compact}]`;
                        } else {
                            this.badgeLeft.innerText = `CLASSIFIED LAND-COVER [${compact}]`;
                        }
                    }
                } else {
                    this.classifiedImageSrc = result.resultBlob;
                    this.confidenceHeatmapSrc = null;

                    if (this.compCanvasBase && this.compCanvasClip) {
                        const container = document.getElementById('comparison-container');
                        const cw = container?.clientWidth || 700;
                        const ch = cw;

                        await drawImageToCanvas(result.originalFile, this.compCanvasBase, cw, ch);
                        await drawImageToCanvas(result.resultBlob, this.compCanvasClip, cw, ch);
                    }

                    if (this.badgeLeft && result.inferenceTimeMs) {
                        const ms = parseFloat(result.inferenceTimeMs);
                        const compact = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
                        this.badgeLeft.innerText = `AI PREDICTED OPTICAL [${compact}]`;
                    }
                }
            } catch (err) {
                console.error('Backend processing failed:', err);
                alert(`Backend Processing Error: ${err.message}`);
                // Fallback to preset simulation on error
                this.drawComparisonData(this.currentPreset);
            } finally {
                this.currentFetchPromise = null;
                this.finishProcessing();
            }
            return;
        }

        // Preset simulation if no file was uploaded
        this.drawComparisonData(this.currentPreset);
        this.finishProcessing();
    }

    // --- Class Coverage Stats Renderer ---
    renderClassStats(classPercentages, meanConfidence) {
        const panel = document.getElementById('class-stats-panel');
        const barsContainer = document.getElementById('class-stats-bars');
        const confLabel = document.getElementById('mean-confidence-label');
        if (!panel || !barsContainer || !classPercentages) return;

        const CLASS_META = [
            { key: 'Water',           color: 'rgb(0, 0, 255)',       hex: '#0000ff' },
            { key: 'Vegetation',      color: 'rgb(0, 200, 0)',       hex: '#00c800' },
            { key: 'Urban/Built-up',  color: 'rgb(150, 150, 150)',   hex: '#969696' },
            { key: 'Bare Soil/Other', color: 'rgb(194, 178, 128)',   hex: '#c2b280' },
        ];

        barsContainer.innerHTML = CLASS_META.map(({ key, hex }) => {
            const pct = classPercentages[key] ?? 0;
            return `
                <div class="stat-row">
                    <div class="stat-swatch" style="background:${hex}"></div>
                    <span class="stat-label">${key}</span>
                    <div class="stat-bar-wrap">
                        <div class="stat-bar-fill" style="width:${pct}%;background:${hex}aa"></div>
                    </div>
                    <span class="stat-pct">${pct.toFixed(1)}%</span>
                </div>
            `;
        }).join('');

        if (confLabel && meanConfidence !== null) {
            confLabel.textContent = `Confidence: ${(meanConfidence * 100).toFixed(1)}%`;
        }

        panel.classList.remove('hidden');
    }

    finishProcessing() {
        this.isProcessing = false;

        // Restore interactive controls
        if (this.btnRunModel) {
            this.btnRunModel.disabled = false;
            if (this.btnRunModel.dataset.origText) {
                this.btnRunModel.innerHTML = this.btnRunModel.dataset.origText;
            }
        }
        if (this.dropzone) {
            this.dropzone.style.pointerEvents = 'auto';
            this.dropzone.style.opacity = '1.0';
        }
    }

    resetFlow() {
        this.isProcessing = false;
        if (this.btnRunModel) {
            this.btnRunModel.disabled = false;
        }
        if (this.dropzone) {
            this.dropzone.style.pointerEvents = 'auto';
            this.dropzone.style.opacity = '1.0';
        }
    }

    // --- Change Detection Handler ---
    async processChangeDetection() {
        if (!this.lastBeforeFile || !this.lastAfterFile) {
            alert('Please upload both Before and After SAR rasters to run Change Detection.');
            return;
        }

        if (this.isProcessing) return;
        this.isProcessing = true;

        if (this.btnRunModel) {
            this.btnRunModel.disabled = true;
            this.btnRunModel.dataset.origText = this.btnRunModel.innerHTML;
            this.btnRunModel.innerHTML = `
                <svg class="spin" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
                ANALYZING CHANGES...
            `;
        }

        try {
            const filterSelect = document.getElementById('filter-select');
            const filterType = filterSelect ? filterSelect.value : 'enhanced_lee';

            const formData = new FormData();
            formData.append('before_file', this.lastBeforeFile);
            formData.append('after_file', this.lastAfterFile);
            
            const pxRes = this.pixelResolutionInput ? parseFloat(this.pixelResolutionInput.value) || 10.0 : 10.0;
            formData.append('pixel_resolution_m', pxRes.toString());
            formData.append('filter_type', filterType);

            const response = await fetch('http://localhost:8000/change-detect', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(errData.detail || 'Change detection failed.');
            }

            const data = await response.json();

            // Hide standard comparison container and show change stage wrapper
            if (this.comparisonContainer) this.comparisonContainer.classList.add('hidden');
            if (this.compareTriple) this.compareTriple.classList.add('hidden');
            if (this.legend) this.legend.classList.add('hidden');
            if (this.changeStageWrapper) this.changeStageWrapper.classList.remove('hidden');

            // Update headline text
            if (this.changeHeadlineText) {
                this.changeHeadlineText.textContent = `${data.percent_changed}% of the region changed (${data.total_area_km2} km² total)`;
            }

            // Cache stats for Narrative generation
            this.lastChangeStats = {
                transitions: data.transitions || [],
                total_area_km2: data.total_area_km2 || 0.0,
                percent_changed: data.percent_changed || 0.0
            };

            // Draw canvases
            const panelW = 350;
            const panelH = 350;
            if (this.changeCanvasBefore) await drawImageToCanvas(data.before_classified_image, this.changeCanvasBefore, panelW, panelH);
            if (this.changeCanvasAfter) await drawImageToCanvas(data.after_classified_image, this.changeCanvasAfter, panelW, panelH);
            if (this.changeCanvasOverlay) await drawImageToCanvas(data.change_overlay_image, this.changeCanvasOverlay, panelW, panelH);

            // Populate transitions summary table
            if (this.changeTableBody) {
                if (!data.transitions || data.transitions.length === 0) {
                    this.changeTableBody.innerHTML = '<tr><td colspan="3" style="text-align:center; color:var(--accent-teal);">No land-cover class transitions detected between rasters.</td></tr>';
                } else {
                    this.changeTableBody.innerHTML = data.transitions.map(t => `
                        <tr>
                            <td><strong>${t.from_class}</strong> → <strong>${t.to_class}</strong></td>
                            <td style="font-family:var(--font-mono); font-weight:700; color:var(--accent-cyan);">${t.area_km2} km²</td>
                            <td style="font-family:var(--font-mono); font-weight:600; color:var(--text-white);">${t.percent_of_image}%</td>
                        </tr>
                    `).join('');
                }
            }

        } catch (err) {
            console.error('Change Detection failed:', err);
            alert(`Change Detection Error: ${err.message}`);
        } finally {
            this.isProcessing = false;
            if (this.btnRunModel) {
                this.btnRunModel.disabled = false;
                if (this.btnRunModel.dataset.origText) {
                    this.btnRunModel.innerHTML = this.btnRunModel.dataset.origText;
                }
            }
        }
    }

    // --- Multi-Sensor Fusion Handler ---
    async processFusion() {
        if (!this.lastFusionSarFile || !this.lastFusionOptFile) {
            alert('Please upload both Primary SAR Raster and Partial Optical Reference files for Fusion.');
            return;
        }

        if (this.isProcessing) return;
        this.isProcessing = true;

        if (this.btnRunModel) {
            this.btnRunModel.disabled = true;
            this.btnRunModel.dataset.origText = this.btnRunModel.innerHTML;
            this.btnRunModel.innerHTML = `
                <svg class="spin" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
                FUSING SENSORS...
            `;
        }

        try {
            const filterSelect = document.getElementById('filter-select');
            const filterType = filterSelect ? filterSelect.value : 'enhanced_lee';

            const formData = new FormData();
            formData.append('sar_file', this.lastFusionSarFile);
            formData.append('optical_file', this.lastFusionOptFile);
            formData.append('filter_type', filterType);

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000);

            const response = await fetch('http://localhost:8000/fuse', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (status ${response.status})`);
            }

            const imageBlob = await response.blob();
            const inferenceTimeMs = response.headers.get('X-Inference-Time-Ms') || 'N/A';

            // Show main comparison stage container
            if (this.comparisonContainer) {
                this.comparisonContainer.classList.remove('hidden');
            }

            const cw = this.comparisonContainer?.clientWidth || 700;
            const ch = cw;

            await drawImageToCanvas(this.lastFusionSarFile, this.compCanvasBase, cw, ch);
            await drawImageToCanvas(imageBlob, this.compCanvasClip, cw, ch);

            if (this.badgeLeft) {
                const ms = parseFloat(inferenceTimeMs);
                const compact = !isNaN(ms) ? (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`) : 'N/A';
                this.badgeLeft.innerText = `FUSED OPTICAL SPECTRUM [${compact}]`;
            }

            this.lastInferenceTimeMs = inferenceTimeMs;
            if (inferenceTimeMs !== 'N/A') {
                localStorage.setItem('aether_last_inference_ms', inferenceTimeMs);
            }

            this.fetchHistory();
        } catch (err) {
            console.error('Fusion failed:', err);
            alert(`Fusion Error: ${err.message}`);
        } finally {
            this.isProcessing = false;
            if (this.btnRunModel) {
                this.btnRunModel.disabled = false;
                if (this.btnRunModel.dataset.origText) {
                    this.btnRunModel.innerHTML = this.btnRunModel.dataset.origText;
                }
            }
        }
    }

    // --- AI Narrative Generation via Groq LLM API ---
    async generateNarrative(mode) {
        const isClassify = mode === 'classify';
        const btn = isClassify ? this.btnNarrateClassify : this.btnNarrateChange;
        const box = isClassify ? this.narrativeBoxClassify : this.narrativeBoxChange;
        const textEl = isClassify ? this.narrativeTextClassify : this.narrativeTextChange;

        if (!btn || !box || !textEl) return;

        let stats = isClassify ? this.lastClassifyStats : this.lastChangeStats;

        if (!stats) {
            if (isClassify) {
                stats = {
                    class_percentages: { "Water": 32.5, "Vegetation": 45.0, "Urban/Built-up": 14.2, "Bare Soil/Other": 8.3 },
                    mean_confidence: 0.88,
                };
            } else {
                stats = {
                    transitions: [
                        { from_class: "Vegetation", to_class: "Urban/Built-up", area_km2: 1.2, percent_of_image: 18.75 },
                        { from_class: "Vegetation", to_class: "Water", area_km2: 0.4, percent_of_image: 6.25 }
                    ],
                    total_area_km2: 6.4,
                    percent_changed: 25.0
                };
            }
        }

        // Show loading state on button
        const origHtml = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `
            <svg class="spin" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
            <span>GENERATE ANALYSIS...</span>
        `;

        try {
            const response = await fetch('http://localhost:8000/narrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: isClassify ? 'classify' : 'change-detect', stats })
            });

            box.classList.remove('hidden');

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                const msg = errData.detail || 'Analysis unavailable';
                textEl.textContent = msg.includes('GROQ_API_KEY') ? 'Analysis unavailable (GROQ_API_KEY not configured)' : 'Analysis unavailable';
                textEl.classList.add('narrative-unavailable');
                return;
            }

            const data = await response.json();
            if (data.narrative) {
                textEl.textContent = data.narrative;
                textEl.classList.remove('narrative-unavailable');
            } else {
                textEl.textContent = 'Analysis unavailable';
                textEl.classList.add('narrative-unavailable');
            }
        } catch (err) {
            console.warn('Narrate call failed:', err);
            box.classList.remove('hidden');
            textEl.textContent = 'Analysis unavailable';
            textEl.classList.add('narrative-unavailable');
        } finally {
            btn.disabled = false;
            btn.innerHTML = origHtml;
        }
    }

    // --- Compare Both ---
    async runCompareBoth() {
        // Need a file to compare — either last uploaded or current preset
        let file = this.lastUploadedFile;

        if (!file) {
            // Try to load current preset as fallback
            const filenameMap = {
                'coastal': 'coastal.png',
                'metropolis': 'metropolis.png',
                'delta': 'river-delta.png',
                'agriculture': 'crop-plots.png'
            };
            const filename = filenameMap[this.currentPreset] || `${this.currentPreset}.png`;
            const samplePath = `sample-images/${filename}`;

            try {
                const res = await fetch(samplePath);
                if (!res.ok) throw new Error(`Could not load sample image at ${samplePath}`);
                const blob = await res.blob();
                file = new File([blob], filename, { type: blob.type || 'image/png' });
                this.lastUploadedFile = file;
            } catch (err) {
                alert('Please upload or select a SAR image first.');
                return;
            }
        }

        if (this.isProcessing) return;
        this.isProcessing = true;

        // Disable Compare button and show loading
        if (this.btnCompareBoth) {
            this.btnCompareBoth.disabled = true;
            this.btnCompareBoth.dataset.origText = this.btnCompareBoth.innerHTML;
            this.btnCompareBoth.innerHTML = `
                <svg class="spin" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
                COMPARING...
            `;
        }

        try {
            // Run both endpoints in parallel
            const [colorizeResult, classifyResult] = await Promise.all([
                processSARFile(file, 'colorize', this.lastGroundTruthFile),
                processSARFile(file, 'classify', null),
            ]);

            // Refresh history after running compare both
            this.fetchHistory();

            // Show PSNR & SSIM metrics if returned
            if (colorizeResult.psnr && colorizeResult.ssim && this.metricsBanner) {
                if (this.metricPsnr) this.metricPsnr.textContent = colorizeResult.psnr;
                if (this.metricSsim) this.metricSsim.textContent = colorizeResult.ssim;
                this.metricsBanner.classList.remove('hidden');
            }

            // Show the dual slider comparison container
            this.showCompareTriple();

            const colorizeBaseCanvas = document.getElementById('compare-canvas-colorize-base');
            const colorizeClipCanvas = document.getElementById('compare-canvas-colorize-clip');
            const classifyBaseCanvas = document.getElementById('compare-canvas-classify-base');
            const classifyClipCanvas = document.getElementById('compare-canvas-classify-clip');

            if (colorizeBaseCanvas && colorizeClipCanvas && classifyBaseCanvas && classifyClipCanvas) {
                const panelW = colorizeBaseCanvas.parentElement?.clientWidth || 400;
                const panelH = panelW; // 1:1 aspect ratio square

                const classifySrc = classifyResult.classifiedImage || classifyResult.resultBlob;

                await Promise.all([
                    // Panel 1: SAR Input (Base) vs Colorization (Clip)
                    drawImageToCanvas(file, colorizeBaseCanvas, panelW, panelH),
                    drawImageToCanvas(colorizeResult.resultBlob, colorizeClipCanvas, panelW, panelH),

                    // Panel 2: SAR Input (Base) vs Classification (Clip)
                    drawImageToCanvas(file, classifyBaseCanvas, panelW, panelH),
                    drawImageToCanvas(classifySrc, classifyClipCanvas, panelW, panelH),
                ]);

                // Update panel labels with inference timing
                const colorizeLabel = document.getElementById('compare-label-colorize');
                const classifyLabel = document.getElementById('compare-label-classify');

                if (colorizeLabel && colorizeResult.inferenceTimeMs) {
                    const ms = parseFloat(colorizeResult.inferenceTimeMs);
                    const compact = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
                    colorizeLabel.textContent = `REALISTIC COLORIZATION VS SAR [${compact}]`;
                }
                if (classifyLabel && classifyResult.inferenceTimeMs) {
                    const ms = parseFloat(classifyResult.inferenceTimeMs);
                    const compact = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
                    classifyLabel.textContent = `CLASSIFIED LAND-COVER VS SAR [${compact}]`;
                }
            }
        } catch (err) {
            console.error('Compare Both failed:', err);
            alert(`Compare Both Error: ${err.message}`);
        } finally {
            this.isProcessing = false;
            if (this.btnCompareBoth) {
                this.btnCompareBoth.disabled = false;
                if (this.btnCompareBoth.dataset.origText) {
                    this.btnCompareBoth.innerHTML = this.btnCompareBoth.dataset.origText;
                }
            }
        }
    }

    showCompareTriple() {
        if (this.compareTriple) {
            this.compareTriple.classList.remove('hidden');
        }
    }

    hideCompareTriple() {
        if (this.compareTriple) {
            this.compareTriple.classList.add('hidden');
        }
    }

    // --- Generate Analysis Report PNG ---
    async generateAnalysisReport() {
        if (this.currentMode === 'change_detect') {
            if (!this.changeCanvasBefore || !this.changeCanvasAfter || !this.changeCanvasOverlay) {
                alert('No Change Detection results available to generate report.');
                return;
            }

            const now = new Date();
            const timestampStr = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';

            const reportCanvas = document.createElement('canvas');
            reportCanvas.width = 1200;
            reportCanvas.height = 1000;
            const ctx = reportCanvas.getContext('2d');

            // Background
            const bgGrad = ctx.createLinearGradient(0, 0, 0, 1000);
            bgGrad.addColorStop(0, '#030611');
            bgGrad.addColorStop(1, '#080e1c');
            ctx.fillStyle = bgGrad;
            ctx.fillRect(0, 0, 1200, 1000);

            // Frame border
            ctx.strokeStyle = 'rgba(255, 0, 160, 0.4)';
            ctx.lineWidth = 4;
            ctx.strokeRect(16, 16, 1168, 968);

            // Title
            ctx.fillStyle = '#ff00a0';
            ctx.font = 'bold 26px "Outfit", sans-serif';
            ctx.fillText('AETHER-SAR | TEMPORAL CHANGE DETECTION REPORT', 45, 65);

            ctx.fillStyle = '#64748b';
            ctx.font = '14px "JetBrains Mono", monospace';
            ctx.fillText(`Generated: ${timestampStr}`, 820, 65);

            ctx.fillStyle = '#94a3b8';
            ctx.font = '14px "Inter", sans-serif';
            ctx.fillText('Multi-Temporal Spaceborne SAR Pixel-Wise Land-Cover Class Transition Analysis', 45, 95);

            ctx.strokeStyle = 'rgba(255, 0, 160, 0.2)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(45, 110);
            ctx.lineTo(1155, 110);
            ctx.stroke();

            // Headline Stat Banner
            ctx.fillStyle = 'rgba(255, 0, 160, 0.1)';
            ctx.fillRect(45, 130, 1110, 55);
            ctx.strokeStyle = 'rgba(255, 0, 160, 0.3)';
            ctx.strokeRect(45, 130, 1110, 55);

            ctx.fillStyle = '#ff00a0';
            ctx.font = 'bold 16px "Outfit", sans-serif';
            const headline = this.changeHeadlineText ? this.changeHeadlineText.textContent : 'Temporal Change Detection Analysis';
            ctx.fillText(headline, 65, 164);

            // 3-Panel Viewports (350 x 350 each)
            const imgY = 205;
            const imgW = 350;
            const imgH = 350;

            // Viewport 1: Before
            ctx.fillStyle = 'rgba(2, 4, 10, 0.9)';
            ctx.fillRect(45, imgY, imgW, imgH);
            ctx.strokeStyle = 'rgba(0, 243, 255, 0.3)';
            ctx.strokeRect(45, imgY, imgW, imgH);
            ctx.drawImage(this.changeCanvasBefore, 45, imgY, imgW, imgH);

            ctx.fillStyle = 'rgba(0, 243, 255, 0.2)';
            ctx.fillRect(55, imgY + 12, 220, 26);
            ctx.fillStyle = '#00f3ff';
            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            ctx.fillText('BEFORE CLASSIFIED (T1)', 67, imgY + 29);

            // Viewport 2: After
            ctx.fillStyle = 'rgba(2, 4, 10, 0.9)';
            ctx.fillRect(425, imgY, imgW, imgH);
            ctx.strokeStyle = 'rgba(255, 0, 160, 0.3)';
            ctx.strokeRect(425, imgY, imgW, imgH);
            ctx.drawImage(this.changeCanvasAfter, 425, imgY, imgW, imgH);

            ctx.fillStyle = 'rgba(255, 0, 160, 0.2)';
            ctx.fillRect(435, imgY + 12, 210, 26);
            ctx.fillStyle = '#ff00a0';
            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            ctx.fillText('AFTER CLASSIFIED (T2)', 447, imgY + 29);

            // Viewport 3: Overlay Map
            ctx.fillStyle = 'rgba(2, 4, 10, 0.9)';
            ctx.fillRect(805, imgY, imgW, imgH);
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
            ctx.strokeRect(805, imgY, imgW, imgH);
            ctx.drawImage(this.changeCanvasOverlay, 805, imgY, imgW, imgH);

            ctx.fillStyle = 'rgba(0, 255, 136, 0.2)';
            ctx.fillRect(815, imgY + 12, 200, 26);
            ctx.fillStyle = '#00ff88';
            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            ctx.fillText('CHANGE OVERLAY MAP', 827, imgY + 29);

            // Bottom Section: Transitions Summary Palette & Table
            const bottomY = 575;
            ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
            ctx.fillRect(45, bottomY, 1110, 360);
            ctx.strokeStyle = 'rgba(255, 0, 160, 0.25)';
            ctx.strokeRect(45, bottomY, 1110, 360);

            ctx.fillStyle = '#ff00a0';
            ctx.font = 'bold 13px "JetBrains Mono", monospace';
            ctx.fillText('LAND-COVER TRANSITIONS PALETTE & METRICS SUMMARY', 65, bottomY + 30);

            // Swatches Row
            const transitionSwatches = [
                { color: 'rgb(255, 0, 100)', label: 'Veg → Urban' },
                { color: 'rgb(0, 200, 255)', label: 'Veg → Water' },
                { color: 'rgb(255, 140, 0)', label: 'Veg → Bare Soil' },
                { color: 'rgb(0, 255, 128)', label: 'Water → Veg' },
                { color: 'rgb(255, 220, 0)', label: 'Water → Bare Soil' },
                { color: 'rgb(200, 50, 255)', label: 'Bare → Urban' },
            ];

            transitionSwatches.forEach((item, idx) => {
                const sx = 65 + idx * 175;
                const sy = bottomY + 50;

                ctx.fillStyle = item.color;
                ctx.fillRect(sx, sy, 18, 18);
                ctx.strokeStyle = 'rgba(255,255,255,0.4)';
                ctx.strokeRect(sx, sy, 18, 18);

                ctx.fillStyle = '#f8fafc';
                ctx.font = '12px "Inter", sans-serif';
                ctx.fillText(item.label, sx + 26, sy + 14);
            });

            // Table Header in Report
            const tableY = bottomY + 95;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.fillRect(65, tableY, 1070, 28);
            ctx.fillStyle = '#94a3b8';
            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            ctx.fillText('TRANSITION PAIR', 80, tableY + 18);
            ctx.fillText('AREA CHANGED (KM²)', 500, tableY + 18);
            ctx.fillText('PERCENT OF IMAGE AREA', 850, tableY + 18);

            // Table Rows
            const rows = this.changeTableBody ? Array.from(this.changeTableBody.querySelectorAll('tr')) : [];
            let rowY = tableY + 50;

            rows.slice(0, 6).forEach((tr, rIdx) => {
                const tds = tr.querySelectorAll('td');
                if (tds.length === 3) {
                    ctx.fillStyle = rIdx % 2 === 0 ? 'rgba(255, 255, 255, 0.02)' : 'transparent';
                    ctx.fillRect(65, rowY - 18, 1070, 28);

                    ctx.fillStyle = '#ffffff';
                    ctx.font = '13px "Inter", sans-serif';
                    ctx.fillText(tds[0].innerText, 80, rowY);

                    ctx.fillStyle = '#00f3ff';
                    ctx.font = 'bold 13px "JetBrains Mono", monospace';
                    ctx.fillText(tds[1].innerText, 500, rowY);

                    ctx.fillStyle = '#ffffff';
                    ctx.font = '13px "JetBrains Mono", monospace';
                    ctx.fillText(tds[2].innerText, 850, rowY);

                    rowY += 34;
                }
            });

            // Footer Bar
            ctx.fillStyle = '#64748b';
            ctx.font = '12px "Inter", sans-serif';
            ctx.fillText('AETHER-SAR Remote Sensing Suite | Powered by PyTorch UNet Change Detection Engine', 45, 970);
            ctx.fillText('Official Temporal Change Analysis Artifact — Confidential Geospatial Data', 720, 970);

            // Download PNG file
            const timestampFilename = now.toISOString().replace(/[-:]/g, '').replace('T', '_').substring(0, 15);
            const link = document.createElement('a');
            link.download = `AETHER_SAR_CHANGE_DETECTION_REPORT_${timestampFilename}.png`;
            link.href = reportCanvas.toDataURL('image/png');
            link.click();
            return;
        }

        if (!this.compCanvasBase || !this.compCanvasClip) {
            alert('No analysis results available to generate report.');
            return;
        }

        const isClassify = this.currentMode === 'classify';
        const isConfidence = isClassify && this.confidenceCheckbox && this.confidenceCheckbox.checked;

        let modeTitle = 'REALISTIC OPTICAL COLORIZATION';
        if (isClassify) {
            modeTitle = isConfidence ? 'CLASSIFIED CONFIDENCE HEATMAP' : 'CLASSIFIED LAND-COVER MAP';
        }

        const now = new Date();
        const timestampStr = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';

        // Offscreen report canvas layout (1200 x 950 high resolution)
        const reportCanvas = document.createElement('canvas');
        reportCanvas.width = 1200;
        reportCanvas.height = 950;
        const ctx = reportCanvas.getContext('2d');

        // Background gradient
        const bgGrad = ctx.createLinearGradient(0, 0, 0, 950);
        bgGrad.addColorStop(0, '#030611');
        bgGrad.addColorStop(1, '#080e1c');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, 1200, 950);

        // Border frame
        ctx.strokeStyle = 'rgba(0, 243, 255, 0.4)';
        ctx.lineWidth = 4;
        ctx.strokeRect(16, 16, 1168, 918);

        // Title Header
        ctx.fillStyle = '#00f3ff';
        ctx.font = 'bold 26px "Outfit", sans-serif';
        ctx.fillText('AETHER-SAR | REMOTE SENSING ANALYSIS REPORT', 45, 65);

        ctx.fillStyle = '#64748b';
        ctx.font = '14px "JetBrains Mono", monospace';
        ctx.fillText(`Generated: ${timestampStr}`, 820, 65);

        // Subtitle & Divider Line
        ctx.fillStyle = '#94a3b8';
        ctx.font = '14px "Inter", sans-serif';
        ctx.fillText('Spaceborne Synthetic Aperture Radar (SAR) Deep Learning Spectrum Synthesis', 45, 95);

        ctx.strokeStyle = 'rgba(0, 243, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(45, 110);
        ctx.lineTo(1155, 110);
        ctx.stroke();

        // Metadata Telemetry Cards Bar
        const cardY = 130;
        const cardH = 75;

        // Card 1: Mode
        ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
        ctx.fillRect(45, cardY, 260, cardH);
        ctx.strokeStyle = 'rgba(0, 243, 255, 0.2)';
        ctx.strokeRect(45, cardY, 260, cardH);
        ctx.fillStyle = '#64748b';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('ANALYSIS MODE', 60, cardY + 25);
        ctx.fillStyle = '#00f3ff';
        ctx.font = 'bold 13px "Inter", sans-serif';
        ctx.fillText(modeTitle, 60, cardY + 52);

        // Card 2: Inference Speed
        ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
        ctx.fillRect(325, cardY, 260, cardH);
        ctx.strokeRect(325, cardY, 260, cardH);
        ctx.fillStyle = '#64748b';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('INFERENCE SPEED', 340, cardY + 25);
        ctx.fillStyle = '#00ff88';
        ctx.font = 'bold 15px "JetBrains Mono", monospace';
        const speedText = this.lastInferenceTimeMs ? `${parseFloat(this.lastInferenceTimeMs).toFixed(1)} ms` : 'FastAPI GPU';
        ctx.fillText(speedText, 340, cardY + 52);

        // Card 3: Checkpoint Model
        ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
        ctx.fillRect(605, cardY, 260, cardH);
        ctx.strokeRect(605, cardY, 260, cardH);
        ctx.fillStyle = '#64748b';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('AI MODEL CHECKPOINT', 620, cardY + 25);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px "JetBrains Mono", monospace';
        ctx.fillText(this.lastModelCheckpoint, 620, cardY + 52);

        // Card 4: Sensor Modality
        ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
        ctx.fillRect(885, cardY, 270, cardH);
        ctx.strokeRect(885, cardY, 270, cardH);
        ctx.fillStyle = '#64748b';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('INPUT SENSOR', 900, cardY + 25);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px "Inter", sans-serif';
        ctx.fillText('Sentinel-1A (C-Band SAR)', 900, cardY + 52);

        // Image Viewports Side-by-Side (520 x 420 each)
        const imgY = 230;
        const imgW = 535;
        const imgH = 430;

        // Image 1: Raw SAR Input
        ctx.fillStyle = 'rgba(2, 4, 10, 0.9)';
        ctx.fillRect(45, imgY, imgW, imgH);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.strokeRect(45, imgY, imgW, imgH);

        // Draw base SAR canvas
        ctx.drawImage(this.compCanvasBase, 45, imgY, imgW, imgH);

        // Image 1 Label Badge
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fillRect(55, imgY + 12, 310, 28);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.strokeRect(55, imgY + 12, 310, 28);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        ctx.fillText('RAW SAR MONOCHROMATIC BACKSCATTER', 67, imgY + 31);

        // Image 2: Model Output
        ctx.fillStyle = 'rgba(2, 4, 10, 0.9)';
        ctx.fillRect(620, imgY, imgW, imgH);
        ctx.strokeStyle = 'rgba(0, 243, 255, 0.3)';
        ctx.strokeRect(620, imgY, imgW, imgH);

        // Draw model output canvas
        ctx.drawImage(this.compCanvasClip, 620, imgY, imgW, imgH);

        // Image 2 Label Badge
        ctx.fillStyle = 'rgba(0, 243, 255, 0.2)';
        ctx.fillRect(630, imgY + 12, 320, 28);
        ctx.strokeStyle = '#00f3ff';
        ctx.strokeRect(630, imgY + 12, 320, 28);
        ctx.fillStyle = '#00f3ff';
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        ctx.fillText(modeTitle, 642, imgY + 31);

        // Bottom Section: Legend or Evaluation Metrics
        const bottomY = 685;

        if (isClassify) {
            // Draw Classification Legend Swatches in Report
            ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
            ctx.fillRect(45, bottomY, 1110, 100);
            ctx.strokeStyle = 'rgba(0, 243, 255, 0.25)';
            ctx.strokeRect(45, bottomY, 1110, 100);

            ctx.fillStyle = '#00f3ff';
            ctx.font = 'bold 12px "JetBrains Mono", monospace';
            ctx.fillText('LAND-COVER CLASSIFICATION PALETTE LEGEND', 65, bottomY + 30);

            const legendSwatches = [
                { color: 'rgb(0, 0, 255)', label: 'Water (Class 0)' },
                { color: 'rgb(0, 200, 0)', label: 'Vegetation (Class 1)' },
                { color: 'rgb(150, 150, 150)', label: 'Urban / Built-up (Class 2)' },
                { color: 'rgb(194, 178, 128)', label: 'Bare Soil / Other (Class 3)' },
            ];

            legendSwatches.forEach((item, idx) => {
                const sx = 65 + idx * 260;
                const sy = bottomY + 50;

                ctx.fillStyle = item.color;
                ctx.fillRect(sx, sy, 22, 22);
                ctx.strokeStyle = 'rgba(255,255,255,0.4)';
                ctx.strokeRect(sx, sy, 22, 22);

                ctx.fillStyle = '#f8fafc';
                ctx.font = '13px "Inter", sans-serif';
                ctx.fillText(item.label, sx + 32, sy + 16);
            });
        } else if (this.lastPsnr && this.lastSsim) {
            // Draw Ground-Truth Evaluation Metrics Box in Report
            ctx.fillStyle = 'rgba(0, 255, 136, 0.08)';
            ctx.fillRect(45, bottomY, 1110, 85);
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.4)';
            ctx.strokeRect(45, bottomY, 1110, 85);

            ctx.fillStyle = '#00ff88';
            ctx.font = 'bold 13px "JetBrains Mono", monospace';
            ctx.fillText('GROUND-TRUTH OPTICAL EVALUATION ACCURACY METRICS', 65, bottomY + 32);

            ctx.fillStyle = '#f8fafc';
            ctx.font = '16px "Inter", sans-serif';
            ctx.fillText(`Peak Signal-to-Noise Ratio (PSNR): `, 65, bottomY + 62);
            ctx.fillStyle = '#00ff88';
            ctx.font = 'bold 18px "JetBrains Mono", monospace';
            ctx.fillText(`${this.lastPsnr} dB`, 335, bottomY + 62);

            ctx.fillStyle = '#f8fafc';
            ctx.font = '16px "Inter", sans-serif';
            ctx.fillText(`Structural Similarity Index (SSIM): `, 520, bottomY + 62);
            ctx.fillStyle = '#00ff88';
            ctx.font = 'bold 18px "JetBrains Mono", monospace';
            ctx.fillText(`${this.lastSsim}`, 815, bottomY + 62);
        }

        // Footer Bar
        ctx.fillStyle = '#64748b';
        ctx.font = '12px "Inter", sans-serif';
        ctx.fillText('AETHER-SAR Remote Sensing Suite | Powered by PyTorch & U-Net GAN Architecture', 45, 915);
        ctx.fillText('Official Analysis Artifact — Confidential Geospatial Remote Sensing Data', 750, 915);

        // Download PNG file
        const timestampFilename = now.toISOString().replace(/[-:]/g, '').replace('T', '_').substring(0, 15);
        const link = document.createElement('a');
        link.download = `AETHER_SAR_ANALYSIS_REPORT_${timestampFilename}.png`;
        link.href = reportCanvas.toDataURL('image/png');
        link.click();
    }

    // --- Canvas Setup & Drawing ---
    setupComparisonCanvases() {
        if (!this.compCanvasBase || !this.compCanvasClip) return;
        this.compCanvasBase.width = this.compCanvasClip.width = 560;
        this.compCanvasBase.height = this.compCanvasClip.height = 560;
        this.drawComparisonData('coastal');
    }

    drawComparisonData(presetKey) {
        const ctxBase = this.compCanvasBase.getContext('2d');
        const ctxClip = this.compCanvasClip.getContext('2d');
        const w = 560, h = 560;

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
        const isConfidence = this.confidenceCheckbox && this.confidenceCheckbox.checked;
        let suffix = 'COLORIZED';
        if (this.currentMode === 'classify') {
            suffix = isConfidence ? 'CONFIDENCE_HEATMAP' : 'CLASSIFIED';
        }

        const link = document.createElement('a');
        link.download = `SAR_${suffix}_${this.currentPreset.toUpperCase()}_HD.png`;
        link.href = this.compCanvasClip.toDataURL('image/png');
        link.click();
    }
}

/**
 * Parameterized helper: Binds a range input element to control the clip width of a curtain wrapper element.
 */
function bindCurtainSlider(sliderInputEl, clipWrapperEl) {
    if (!sliderInputEl || !clipWrapperEl) return;
    sliderInputEl.addEventListener('input', (e) => {
        const val = e.target.value;
        clipWrapperEl.style.width = `${val}%`;
        const parent = sliderInputEl.parentElement;
        const handle = parent ? parent.querySelector('.curtain-handle') : null;
        if (handle) {
            handle.style.left = `${val}%`;
        }
    });
}

/**
 * Calculates human-readable relative time string (e.g. "2m ago", "1h ago", "Just now").
 */
function getRelativeTimeString(isoTimestamp) {
    if (!isoTimestamp) return 'Just now';
    const date = new Date(isoTimestamp);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);

    if (diffSec < 10) return 'Just now';
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
}

window.addEventListener('DOMContentLoaded', () => {
    window.sarUI = new SARUIController();
});

/**
 * Sends an uploaded SAR image file (and optional Ground-Truth file) to FastAPI backend.
 */
async function processSARFile(file, mode = 'colorize', groundTruthFile = null) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    const endpoint = mode === 'classify' ? 'classify' : 'colorize';

    try {
        const filterSelect = document.getElementById('filter-select');
        const filterType = filterSelect ? filterSelect.value : 'enhanced_lee';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('filter_type', filterType);

        if (groundTruthFile) {
            formData.append('ground_truth', groundTruthFile);
        }

        const response = await fetch(`http://localhost:8000/${endpoint}`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (status ${response.status})`);
        }

        const psnr = response.headers.get('X-PSNR');
        const ssim = response.headers.get('X-SSIM');

        if (mode === 'classify') {
            // Parses JSON response containing both classified image & confidence heatmap base64 data
            const data = await response.json();
            return {
                originalFile: file,
                classifiedImage: data.classified_image,
                confidenceHeatmap: data.confidence_heatmap,
                inferenceTimeMs: data.inference_time_ms || response.headers.get('X-Inference-Time-Ms') || 'N/A',
                imageSize: data.image_size || '',
                modelCheckpoint: data.model_checkpoint || 'classifier',
                classPercentages: data.class_percentages || null,
                meanConfidence: data.mean_confidence ?? null,
                mode: mode,
                psnr: psnr,
                ssim: ssim,
            };
        } else {
            // PNG blob response for colorize
            const imageBlob = await response.blob();
            const inferenceTimeMs = response.headers.get('X-Inference-Time-Ms');
            const imageSize = response.headers.get('X-Image-Size');
            const modelCheckpoint = response.headers.get('X-Model-Checkpoint');

            return {
                originalFile: file,
                resultBlob: imageBlob,
                inferenceTimeMs: inferenceTimeMs || 'N/A',
                imageSize: imageSize || '',
                modelCheckpoint: modelCheckpoint || '',
                mode: mode,
                psnr: psnr,
                ssim: ssim,
            };
        }
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Backend request timed out after 30 seconds.');
        }
        throw error;
    }
}

/**
 * Renders an image file/blob/dataURL onto a target canvas, scaled proportionally.
 */
function drawImageToCanvas(src, canvas, containerWidth, containerHeight) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const cw = containerWidth || 600;
            const ch = containerHeight || 360;
            canvas.width = cw;
            canvas.height = ch;

            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, cw, ch);

            // Scale proportionally to fit, preserving aspect ratio (letterboxing)
            const scale = Math.min(cw / img.width, ch / img.height);
            const dw = img.width * scale;
            const dh = img.height * scale;
            const dx = (cw - dw) / 2;
            const dy = (ch - dh) / 2;

            ctx.drawImage(img, dx, dy, dw, dh);
            resolve();
        };
        img.onerror = reject;
        img.src = typeof src === 'string' ? src : URL.createObjectURL(src);
    });
}
