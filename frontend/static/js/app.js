// FLUX API ComfyUI Frontend
class FluxAPI {
    constructor() {
        this.currentService = 'fp4';
        this.currentPort = '8000';
        // Build base host without an existing port to avoid patterns like :8000:8001
        this.hostBase = window.location.protocol + '//' + window.location.hostname;
        this.isGenerating = false;
        this.generationHistory = [];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSliders();
        this.checkServiceStatus();
        this.loadHistory();
        
        // Check service status every 30 seconds
        setInterval(() => this.checkServiceStatus(), 30000);
    }

    setupEventListeners() {
        // Service selection
        document.querySelectorAll('.service-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchService(e.target.closest('.service-btn')));
        });

        // Generate button
        document.getElementById('generate-btn').addEventListener('click', () => this.generateImage());

        // Random seed button
        document.getElementById('random-seed').addEventListener('click', () => this.randomSeed());

        // Clear history
        document.getElementById('clear-history').addEventListener('click', () => this.clearHistory());

        // Modal controls
        document.getElementById('close-modal').addEventListener('click', () => this.closeModal());
        document.getElementById('download-image').addEventListener('click', () => this.downloadCurrentImage());
        document.getElementById('copy-prompt').addEventListener('click', () => this.copyCurrentPrompt());

        // Close modal on backdrop click
        document.getElementById('image-modal').addEventListener('click', (e) => {
            if (e.target.id === 'image-modal') {
                this.closeModal();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.generateImage();
                } else if (e.key === 'r') {
                    e.preventDefault();
                    this.randomSeed();
                }
            }
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    setupSliders() {
        // Steps slider
        const stepsSlider = document.getElementById('steps');
        const stepsValue = document.getElementById('steps-value');
        stepsSlider.addEventListener('input', (e) => {
            stepsValue.textContent = e.target.value;
        });

        // Guidance slider
        const guidanceSlider = document.getElementById('guidance');
        const guidanceValue = document.getElementById('guidance-value');
        guidanceSlider.addEventListener('input', (e) => {
            guidanceValue.textContent = parseFloat(e.target.value).toFixed(1);
        });

        // LoRA weight slider
        const loraSlider = document.getElementById('lora-weight');
        const loraValue = document.getElementById('lora-weight-value');
        loraSlider.addEventListener('input', (e) => {
            loraValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
    }

    async checkServiceStatus() {
        const statusDot = document.getElementById('service-status');
        const statusText = document.getElementById('status-text');
        
        try {
            const response = await fetch(`${this.hostBase}:${this.currentPort}/health`);
            if (response.ok) {
                statusDot.className = 'status-dot online';
                statusText.textContent = `${this.currentService.toUpperCase()} Online`;
            } else {
                throw new Error('Service offline');
            }
        } catch (error) {
            statusDot.className = 'status-dot offline';
            statusText.textContent = `${this.currentService.toUpperCase()} Offline`;
        }
    }

    switchService(btn) {
        // Update UI
        document.querySelectorAll('.service-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update service config
        this.currentService = btn.dataset.service;
        this.currentPort = btn.dataset.port;
        
        // Check new service status
        this.checkServiceStatus();
    }

    randomSeed() {
        const seedInput = document.getElementById('seed');
        seedInput.value = Math.floor(Math.random() * 4294967295);
    }

    async generateImage() {
        if (this.isGenerating) return;

        const prompt = document.getElementById('prompt').value.trim();
        if (!prompt) {
            this.showError('Please enter a prompt');
            return;
        }

        this.isGenerating = true;
        this.showGenerationStatus(true);
        this.updateGenerateButton(true);

        try {
            const params = this.getGenerationParams();
            
            const response = await fetch(`${this.hostBase}:${this.currentPort}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Generation failed');
            }

            const result = await response.json();
            this.addImageToGallery(result, params);
            this.showSuccess('Image generated successfully!');

        } catch (error) {
            console.error('Generation error:', error);
            this.showError(error.message);
        } finally {
            this.isGenerating = false;
            this.showGenerationStatus(false);
            this.updateGenerateButton(false);
        }
    }

    getGenerationParams() {
        const params = {
            prompt: document.getElementById('prompt').value.trim(),
            num_inference_steps: parseInt(document.getElementById('steps').value),
            guidance_scale: parseFloat(document.getElementById('guidance').value),
            width: parseInt(document.getElementById('width').value),
            height: parseInt(document.getElementById('height').value)
        };

        // Optional parameters
        const negativePrompt = document.getElementById('negative-prompt').value.trim();
        if (negativePrompt) {
            params.negative_prompt = negativePrompt;
        }

        const seed = document.getElementById('seed').value;
        if (seed) {
            params.seed = parseInt(seed);
        }

        const loraName = document.getElementById('lora-name').value.trim();
        if (loraName) {
            params.lora_name = loraName;
        }

        const loraWeight = parseFloat(document.getElementById('lora-weight').value);
        if (loraWeight !== 1.0) {
            params.lora_weight = loraWeight;
        }

        return params;
    }

    addImageToGallery(result, params) {
        // Remove placeholder if exists
        const placeholder = document.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        const gallery = document.getElementById('image-gallery');
        const imageItem = document.createElement('div');
        imageItem.className = 'image-item';
        
        const imageUrl = `${this.hostBase}:${this.currentPort}${result.download_url}`;
        
        imageItem.innerHTML = `
            <img src="${imageUrl}" alt="Generated image" loading="lazy">
            <div class="image-meta">
                <div class="prompt">${params.prompt}</div>
                <div class="details">
                    <span>${params.width}×${params.height}</span>
                    <span>${result.generation_time || 'N/A'}</span>
                </div>
            </div>
        `;

        // Add click handler for modal
        imageItem.addEventListener('click', () => {
            this.showImageModal(result, params, imageUrl);
        });

        // Add to beginning of gallery
        gallery.insertBefore(imageItem, gallery.firstChild);

        // Add to history
        const historyItem = {
            result,
            params,
            imageUrl,
            timestamp: new Date().toISOString()
        };
        this.generationHistory.unshift(historyItem);
        this.saveHistory();

        // Limit history to 50 items
        if (this.generationHistory.length > 50) {
            this.generationHistory = this.generationHistory.slice(0, 50);
            // Remove excess items from gallery
            const items = gallery.querySelectorAll('.image-item');
            for (let i = 50; i < items.length; i++) {
                items[i].remove();
            }
        }
    }

    showImageModal(result, params, imageUrl) {
        const modal = document.getElementById('image-modal');
        const modalImage = document.getElementById('modal-image');
        const modalPrompt = document.getElementById('modal-prompt');
        const modalParams = document.getElementById('modal-params');
        const modalTime = document.getElementById('modal-time');

        modalImage.src = imageUrl;
        modalPrompt.textContent = params.prompt;
        modalParams.textContent = `${params.width}×${params.height}, ${params.num_inference_steps} steps, guidance: ${params.guidance_scale}`;
        modalTime.textContent = result.generation_time || 'N/A';

        // Store current image data for download/copy
        this.currentModalData = { result, params, imageUrl };

        modal.classList.remove('hidden');
    }

    closeModal() {
        document.getElementById('image-modal').classList.add('hidden');
        this.currentModalData = null;
    }

    async downloadCurrentImage() {
        if (!this.currentModalData) return;

        try {
            const response = await fetch(this.currentModalData.imageUrl);
            const blob = await response.blob();
            
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = this.currentModalData.result.filename || 'generated-image.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);

            this.showSuccess('Image downloaded!');
        } catch (error) {
            this.showError('Failed to download image');
        }
    }

    copyCurrentPrompt() {
        if (!this.currentModalData) return;

        navigator.clipboard.writeText(this.currentModalData.params.prompt).then(() => {
            this.showSuccess('Prompt copied to clipboard!');
        }).catch(() => {
            this.showError('Failed to copy prompt');
        });
    }

    clearHistory() {
        if (confirm('Clear all generated images from history?')) {
            this.generationHistory = [];
            this.saveHistory();
            
            const gallery = document.getElementById('image-gallery');
            gallery.innerHTML = `
                <div class="placeholder">
                    <i class="fas fa-image"></i>
                    <p>Generated images will appear here</p>
                </div>
            `;
        }
    }

    showGenerationStatus(show) {
        const status = document.getElementById('generation-status');
        const message = document.getElementById('status-message');
        
        if (show) {
            message.textContent = 'Generating image...';
            status.classList.remove('hidden');
        } else {
            status.classList.add('hidden');
        }
    }

    updateGenerateButton(generating) {
        const btn = document.getElementById('generate-btn');
        const icon = btn.querySelector('i');
        
        if (generating) {
            btn.disabled = true;
            icon.className = 'fas fa-spinner fa-spin';
            btn.querySelector('span') || (btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...');
        } else {
            btn.disabled = false;
            icon.className = 'fas fa-play';
            btn.innerHTML = '<i class="fas fa-play"></i> Generate Image';
        }
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : 'exclamation-triangle'}"></i>
            <span>${message}</span>
        `;

        // Add styles if not already added
        if (!document.querySelector('#notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                    border-radius: var(--radius);
                    padding: var(--spacing-md);
                    display: flex;
                    align-items: center;
                    gap: var(--spacing-sm);
                    z-index: 1001;
                    box-shadow: 0 4px 12px var(--shadow);
                    transform: translateX(100%);
                    transition: transform 0.3s ease;
                }
                .notification-success {
                    border-left: 4px solid var(--accent-info);
                    color: var(--accent-info);
                }
                .notification-error {
                    border-left: 4px solid var(--accent-secondary);
                    color: var(--accent-secondary);
                }
                .notification.show {
                    transform: translateX(0);
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    saveHistory() {
        try {
            localStorage.setItem('flux-api-history', JSON.stringify(this.generationHistory));
        } catch (error) {
            console.warn('Failed to save history to localStorage:', error);
        }
    }

    loadHistory() {
        try {
            const saved = localStorage.getItem('flux-api-history');
            if (saved) {
                this.generationHistory = JSON.parse(saved);
                this.renderHistoryInGallery();
            }
        } catch (error) {
            console.warn('Failed to load history from localStorage:', error);
        }
    }

    renderHistoryInGallery() {
        const gallery = document.getElementById('image-gallery');
        
        if (this.generationHistory.length === 0) return;

        // Remove placeholder
        const placeholder = document.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        // Render history items
        this.generationHistory.forEach(item => {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';
            
            imageItem.innerHTML = `
                <img src="${item.imageUrl}" alt="Generated image" loading="lazy">
                <div class="image-meta">
                    <div class="prompt">${item.params.prompt}</div>
                    <div class="details">
                        <span>${item.params.width}×${item.params.height}</span>
                        <span>${item.result.generation_time || 'N/A'}</span>
                    </div>
                </div>
            `;

            imageItem.addEventListener('click', () => {
                this.showImageModal(item.result, item.params, item.imageUrl);
            });

            gallery.appendChild(imageItem);
        });
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FluxAPI();
});

// Add some helpful keyboard shortcuts info
console.log(`
🎨 FLUX API Frontend Shortcuts:
Ctrl/Cmd + Enter: Generate image
Ctrl/Cmd + R: Random seed
Escape: Close modal
`);
