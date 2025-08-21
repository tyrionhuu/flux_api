// FLUX API ComfyUI Frontend
class FluxAPI {
    constructor() {
        // Build base host without an existing port to avoid patterns like :8000:8001
        this.hostBase = window.location.protocol + '//' + window.location.hostname;
        this.isGenerating = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        
        // Ensure DOM is fully loaded before setting up sliders
        setTimeout(() => {
            this.setupSliders();
        }, 500);
    }

    setupEventListeners() {
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
        console.log('Setting up sliders...');
        
        // Steps slider
        const stepsSlider = document.getElementById('steps');
        const stepsValue = document.getElementById('steps-value');
        
        console.log('Steps slider element:', stepsSlider);
        console.log('Steps value element:', stepsValue);
        
        if (stepsSlider && stepsValue) {
            // Set initial value
            stepsValue.textContent = stepsSlider.value;
            console.log('Initial steps value set to:', stepsSlider.value);
            
            // Test direct manipulation first
            stepsSlider.oninput = function() {
                console.log('Steps oninput fired, value:', this.value);
                stepsValue.textContent = this.value;
                console.log('Steps value updated to:', this.value);
            };
            
            // Add event listener for real-time updates
            stepsSlider.addEventListener('input', function() {
                console.log('Steps input event fired, value:', this.value);
                stepsValue.textContent = this.value;
                console.log('Steps value updated to:', this.value);
            });
            
            // Also add change event as backup
            stepsSlider.addEventListener('change', function() {
                console.log('Steps change event fired, value:', this.value);
                stepsValue.textContent = this.value;
                console.log('Steps value updated to:', this.value);
            });
            
            console.log('Steps slider initialized with value:', stepsSlider.value);
        } else {
            console.error('Steps slider elements not found');
        }

        // Guidance slider
        const guidanceSlider = document.getElementById('guidance');
        const guidanceValue = document.getElementById('guidance-value');
        
        console.log('Guidance slider element:', guidanceSlider);
        console.log('Guidance value element:', guidanceValue);
        
        if (guidanceSlider && guidanceValue) {
            // Set initial value
            guidanceValue.textContent = parseFloat(guidanceSlider.value).toFixed(1);
            console.log('Initial guidance value set to:', parseFloat(guidanceSlider.value).toFixed(1));
            
            // Test direct manipulation first
            guidanceSlider.oninput = function() {
                console.log('Guidance oninput fired, value:', this.value);
                guidanceValue.textContent = parseFloat(this.value).toFixed(1);
                console.log('Guidance value updated to:', parseFloat(this.value).toFixed(1));
            };
            
            // Add event listener for real-time updates
            guidanceSlider.addEventListener('input', function() {
                console.log('Guidance input event fired, value:', this.value);
                guidanceValue.textContent = parseFloat(this.value).toFixed(1);
                console.log('Guidance value updated to:', parseFloat(this.value).toFixed(1));
            });
            
            // Also add change event as backup
            guidanceSlider.addEventListener('change', function() {
                console.log('Guidance change event fired, value:', this.value);
                guidanceValue.textContent = parseFloat(this.value).toFixed(1);
                console.log('Guidance value updated to:', parseFloat(this.value).toFixed(1));
            });
            
            console.log('Guidance slider initialized with value:', guidanceSlider.value);
        } else {
            console.error('Guidance slider elements not found');
        }
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
            

            
            // Only generate on FP4 model - BF16 disabled
            const response = await fetch(`${this.hostBase}:8000/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(`Generation failed: ${error.detail || 'Unknown error'}`);
            }

            const result = await response.json();
            console.log('Generation completed');
            
            // Show single image without model identification
            this.showSingleImage(result, params);
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
            width: parseInt(document.getElementById('width').value),
            height: parseInt(document.getElementById('height').value)
        };
        


        // Ensure both models use the same seed
        const seed = document.getElementById('seed').value;
        if (seed) {
            params.seed = parseInt(seed);
        } else {
            // Generate a random seed if none specified, so both models use the same one
            params.seed = Math.floor(Math.random() * 4294967295);
        }

        return params;
    }
    


    showSingleImage(result, params) {
        const gallery = document.getElementById('image-gallery');
        
        // Clear previous images
        gallery.innerHTML = '';
        
        const singleItem = document.createElement('div');
        singleItem.className = 'comparison-item';
        
        const imageUrl = `${this.hostBase}:8000${result.download_url}`;
        
        singleItem.innerHTML = `
            <div class="comparison-images">
                <div class="image-container" style="max-width: 1400px; margin: 0 auto;">
                    <img src="${imageUrl}" alt="Generated image" loading="lazy" style="max-width: 100%; height: auto;">
                    <div class="image-meta">
                        <div class="details">
                            <span>${params.width}×${params.height}</span>
                            <span>${result.generation_time || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="comparison-footer">
                <div class="prompt">${params.prompt}</div>
            </div>
        `;

        // Add click handler for the image
        const image = singleItem.querySelector('img');
        image.addEventListener('click', () => {
            this.showImageModal(result, params, imageUrl);
        });

        gallery.appendChild(singleItem);
    }

    addComparisonToGallery(fp4Result, bf16Result, params) {

        const gallery = document.getElementById('image-gallery');
        
        // Clear previous images - replace instead of accumulating
        gallery.innerHTML = '';
        
        const comparisonItem = document.createElement('div');
        comparisonItem.className = 'comparison-item';
        
        const fp4ImageUrl = `${this.hostBase}:8000${fp4Result.download_url}`;
        const bf16ImageUrl = `${this.hostBase}:8001${bf16Result.download_url}`;
        
                        comparisonItem.innerHTML = `
                    <div class="comparison-images">
                                        <div class="image-container">
                            <img src="${fp4ImageUrl}" alt="Generated image" loading="lazy">
                            <div class="image-meta">
                                <div class="details">
                                    <span>${params.width}×${params.height}</span>
                                    <span>${fp4Result.generation_time || 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                        <div class="image-container">
                            <img src="${bf16ImageUrl}" alt="Generated image" loading="lazy">
                            <div class="image-meta">
                                <div class="details">
                                    <span>${params.width}×${params.height}</span>
                                    <span>${bf16Result.generation_time || 'N/A'}</span>
                                </div>
                            </div>
                        </div>
            </div>
            <div class="comparison-footer">
                <div class="prompt">${params.prompt}</div>
            </div>
        `;

        // Add click handlers for individual images
        const fp4Image = comparisonItem.querySelector('.image-container:first-child img');
        const bf16Image = comparisonItem.querySelector('.image-container:last-child img');
        
        fp4Image.addEventListener('click', () => {
            this.showImageModal(fp4Result, params, fp4ImageUrl);
        });
        
        bf16Image.addEventListener('click', () => {
            this.showImageModal(bf16Result, params, bf16ImageUrl);
        });

        // Add the new comparison (replaces previous)
        gallery.appendChild(comparisonItem);
    }



    showImageModal(result, params, imageUrl) {
        const modal = document.getElementById('image-modal');
        const modalImage = document.getElementById('modal-image');
        const modalPrompt = document.getElementById('modal-prompt');
        const modalParams = document.getElementById('modal-params');
        const modalTime = document.getElementById('modal-time');

        modalImage.src = imageUrl;
        modalPrompt.textContent = params.prompt;
        modalParams.textContent = `${params.width}×${params.height}`;
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
        if (confirm('Clear all generated images?')) {
            const gallery = document.getElementById('image-gallery');
            gallery.innerHTML = '';
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
        
        if (generating) {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        } else {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play"></i> Generate';
        }
    }

    showSuccess(message) {
        console.log('✅ Success:', message);
    }

    showError(message) {
        console.error('❌ Error:', message);
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
