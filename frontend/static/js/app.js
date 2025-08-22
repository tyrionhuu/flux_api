// FLUX API ComfyUI Frontend
class FluxAPI {
    constructor() {
        // Use current origin (protocol + host + port) so UI works on any served port
        this.hostBase = window.location.origin;
        this.isGenerating = false;
        this.loraEntries = [];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.addLoraEntry('/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors', 1.0);
        
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
        
        // LoRA controls
        document.getElementById('add-lora').addEventListener('click', () => this.addLoraEntry());
        
        // Apply LoRA button
        const applyLoraBtn = document.getElementById('apply-lora-btn');
        console.log('Apply LoRA button found:', applyLoraBtn);
        if (applyLoraBtn) {
            applyLoraBtn.addEventListener('click', () => {
                console.log('Apply LoRA button clicked!');
                this.showApiCommand();
            });
        } else {
            console.error('Apply LoRA button not found!');
        }

        // Drag-and-drop reordering for LoRA list
        const loraList = document.getElementById('lora-list');
        loraList.addEventListener('dragover', (e) => {
            e.preventDefault();
            const afterElement = this.getDragAfterElement(loraList, e.clientY);
            const dragging = document.querySelector('.lora-entry.dragging');
            if (!dragging) return;
            if (afterElement == null) {
                loraList.appendChild(dragging);
            } else {
                loraList.insertBefore(dragging, afterElement);
            }
        });
        loraList.addEventListener('drop', () => {
            // Re-sync internal order with DOM order
            this.loraEntries = Array.from(loraList.children);
        });

        // Clear history (optional element)
        const clearHistoryBtn = document.getElementById('clear-history');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }
        
        // Modal controls
        document.getElementById('close-modal').addEventListener('click', () => this.closeModal());
        document.getElementById('download-image').addEventListener('click', () => this.downloadCurrentImage());
        document.getElementById('copy-command').addEventListener('click', () => this.copyApiCommand());

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

    addLoraEntry(name = '', weight = 1.0) {
        // Check maximum LoRA limit
        if (this.loraEntries.length >= 3) {
            console.warn('Maximum of 3 LoRAs allowed');
            return;
        }

        const loraList = document.getElementById('lora-list');
        const loraEntry = document.createElement('div');
        loraEntry.className = 'lora-entry';
        loraEntry.setAttribute('draggable', 'true');
        loraEntry.innerHTML = `
            <span class="drag-handle" title="Drag to reorder"><i class="fas fa-grip-vertical"></i></span>
            <input type="text" placeholder="username/model-name or /path/to/lora.safetensors" class="lora-name">
            <input type="number" placeholder="1.0" min="0.0" max="2.0" step="0.1" value="1.0" class="lora-weight">
            <button class="remove-lora" title="Remove LoRA">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Set initial values if provided
        const nameInput = loraEntry.querySelector('.lora-name');
        const weightInput = loraEntry.querySelector('.lora-weight');
        if (name) nameInput.value = name;
        if (typeof weight === 'number') weightInput.value = String(weight);
        
        // Remove functionality
        const removeBtn = loraEntry.querySelector('.remove-lora');
        removeBtn.addEventListener('click', () => this.removeLoraEntry(loraEntry));

        // Drag events
        loraEntry.addEventListener('dragstart', () => {
            loraEntry.classList.add('dragging');
        });
        loraEntry.addEventListener('dragend', () => {
            loraEntry.classList.remove('dragging');
            // Re-sync array with DOM order
            this.loraEntries = Array.from(loraList.children);
        });
        
        loraList.appendChild(loraEntry);
        this.loraEntries.push(loraEntry);
        
        // Update Add LoRA button state
        this.updateAddLoraButtonState();
    }

    updateAddLoraButtonState() {
        const addLoraBtn = document.getElementById('add-lora');
        if (addLoraBtn) {
            if (this.loraEntries.length >= 3) {
                addLoraBtn.disabled = true;
                addLoraBtn.title = 'Maximum of 3 LoRAs reached';
                addLoraBtn.style.opacity = '0.5';
                addLoraBtn.style.cursor = 'not-allowed';
            } else {
                addLoraBtn.disabled = false;
                addLoraBtn.title = 'Add LoRA';
                addLoraBtn.style.opacity = '1';
                addLoraBtn.style.cursor = 'pointer';
            }
        }
    }

    getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('.lora-entry:not(.dragging)')];
        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY, element: null }).element;
    }

    removeLoraEntry(loraEntry) {
        const index = this.loraEntries.indexOf(loraEntry);
        if (index > -1) {
            this.loraEntries.splice(index, 1);
        }
        loraEntry.remove();
        // Ensure the array matches DOM after removal
        const loraList = document.getElementById('lora-list');
        this.loraEntries = Array.from(loraList.children);
        
        // Update Add LoRA button state after removal
        this.updateAddLoraButtonState();
    }

    getLoraConfigs() {
        const configs = [];
        for (const entry of this.loraEntries) {
            const name = entry.querySelector('.lora-name').value.trim();
            const weight = parseFloat(entry.querySelector('.lora-weight').value);
            if (name && !isNaN(weight)) {
                configs.push({ name, weight });
            }
        }
        return configs;
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
            

            
            // Only generate on FP4 model - BF16 disabled; use current origin/port
            const response = await fetch(`${this.hostBase}/generate`, {
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
        
        // Always include LoRA configurations (empty list means remove any applied LoRA)
        params.loras = this.getLoraConfigs();

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
        
        const imageUrl = `${this.hostBase}${result.download_url}`;
        
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



    showApiCommand() {
        const commandSection = document.getElementById('api-command-section');
        const commandElement = document.getElementById('api-command');
        
        // Get current LoRA configuration
        const loras = this.getLoraConfigs();
        const prompt = document.getElementById('prompt').value;
        const width = document.getElementById('width').value;
        const height = document.getElementById('height').value;
        const seed = document.getElementById('seed').value;
        
        // Build the one-liner download command
        let command = `curl -s -X POST "${window.location.origin}/generate" -H "Content-Type: application/json" -d '{"prompt": "${prompt}", "width": ${width}, "height": ${height}`;
        
        if (seed) {
            command += `, "seed": ${seed}`;
        }
        
        if (loras && loras.length > 0) {
            command += `, "loras": [`;
            loras.forEach((lora, index) => {
                command += `{"name": "${lora.name}", "weight": ${lora.weight}}`;
                if (index < loras.length - 1) command += `, `;
            });
            command += `]`;
        }
        
        command += `}' | jq -r '.download_url' | xargs -I {} curl -o "generated_image.png" "${window.location.origin}{}"`;
        
        commandElement.textContent = command;
        commandSection.classList.remove('hidden');
    }

    copyApiCommand() {
        const commandElement = document.getElementById('api-command');
        navigator.clipboard.writeText(commandElement.textContent).then(() => {
            this.showSuccess('API command copied to clipboard!');
        }).catch(() => {
            this.showError('Failed to copy API command');
        });
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
Escape: Close modal
`);
