// FLUX API ComfyUI Frontend
class FluxAPI {
    constructor() {
        // Use current origin (protocol + host + port) so UI works on any served port
        this.hostBase = window.location.origin;
        this.isGenerating = false;
        this.loraEntries = [];
        // LoRA state per new design
        this.appliedLoras = [];
        this.availableLoras = [];
        this.uploadedFiles = new Map();
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableLoras();
    }

    setupEventListeners() {
        // Generate button
        document.getElementById('generate-btn').addEventListener('click', () => this.generateImage());

        // Random seed button
        document.getElementById('random-seed').addEventListener('click', () => this.randomSeed());
        
        // LoRA controls (manual add remains available if button exists)
        const addLoraBtnMain = document.getElementById('add-custom-lora');
        if (addLoraBtnMain) addLoraBtnMain.addEventListener('click', () => this.addCustomLora());
        
        // LoRA file upload
        const uploadLoraBtn = document.getElementById('upload-lora');
        const loraFileInput = document.getElementById('lora-file-input');
        if (uploadLoraBtn) uploadLoraBtn.addEventListener('click', () => this.triggerFileUpload());
        if (loraFileInput) loraFileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        // Add selected LoRA button
        const addSelectedBtn = document.getElementById('add-selected-lora');
        if (addSelectedBtn) {
            addSelectedBtn.addEventListener('click', () => this.addSelectedLora());
        }

        // Clear all LoRAs button
        const clearAllBtn = document.getElementById('clear-all-loras');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => this.clearAllLoras());
        }

        // Refresh LoRAs button
        const refreshBtn = document.getElementById('refresh-loras');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadAvailableLoras();
            });
        }
        
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

        // Load available LoRAs on startup
        this.loadAvailableLoras();

        // Upscaler checkbox
        const upscaleCheckbox = document.getElementById('upscale');
        if (upscaleCheckbox) {
            upscaleCheckbox.addEventListener('change', (e) => {
                const upscaleFactorContainer = document.getElementById('upscale-factor-container');
                if (upscaleFactorContainer) {
                    upscaleFactorContainer.style.display = e.target.checked ? 'block' : 'none';
                }
            });
        }
        
        // Guidance scale slider
        const guidanceScaleSlider = document.getElementById('guidance_scale');
        if (guidanceScaleSlider) {
            guidanceScaleSlider.addEventListener('input', (e) => {
                const valueDisplay = document.getElementById('guidance_scale_value');
                if (valueDisplay) {
                    valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
                }
            });
        }

        // LoRA weight input
        const loraWeightInput = document.getElementById('lora-weight');
        if (loraWeightInput) {
            loraWeightInput.addEventListener('input', () => {
                this.updateApiCommand();
            });
        }
        
        // LoRA info tooltip
        const loraInfoIcon = document.getElementById('lora-info-icon');
        const loraInfoTooltip = document.getElementById('lora-info-tooltip');
        console.log('LoRA info icon found:', loraInfoIcon);
        console.log('LoRA info tooltip found:', loraInfoTooltip);
        console.log('LoRA info icon HTML:', loraInfoIcon ? loraInfoIcon.outerHTML : 'NOT FOUND');
        console.log('LoRA info tooltip HTML:', loraInfoTooltip ? loraInfoTooltip.outerHTML : 'NOT FOUND');
        
        if (loraInfoIcon && loraInfoTooltip) {
            console.log('Setting up LoRA info tooltip event listeners');
            
            // Test if the icon is clickable
            loraInfoIcon.style.cursor = 'pointer';
            console.log('Icon cursor style set to pointer');
            
            loraInfoIcon.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Info icon clicked!');
                console.log('Tooltip before toggle:', loraInfoTooltip.className);
                loraInfoTooltip.classList.toggle('show');
                console.log('Tooltip after toggle:', loraInfoTooltip.className);
                console.log('Tooltip computed styles:', window.getComputedStyle(loraInfoTooltip));
            });
            
            // Close tooltip when clicking outside
            document.addEventListener('click', (e) => {
                if (!loraInfoIcon.contains(e.target) && !loraInfoTooltip.contains(e.target)) {
                    loraInfoTooltip.classList.remove('show');
                }
            });
            
            // Close tooltip when pressing Escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    loraInfoTooltip.classList.remove('show');
                }
            });
        } else {
            console.error('LoRA info elements not found!');
            console.error('Available elements with similar IDs:');
            document.querySelectorAll('[id*="lora"]').forEach(el => {
                console.log('Found element:', el.id, el);
            });
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

    randomSeed() {
        const seedInput = document.getElementById('seed');
        seedInput.value = Math.floor(Math.random() * 4294967295);
    }

    addLoraEntry(name = '', weight = 1.0, isUploaded = false) {
        // Check maximum LoRA limit
        if (this.loraEntries.length >= 3) {
            console.warn('Maximum of 3 LoRAs allowed');
            return;
        }

        const loraList = document.getElementById('lora-list');
        const loraEntry = document.createElement('div');
        loraEntry.className = 'lora-entry';
        loraEntry.setAttribute('draggable', 'true');
        
        // Add special class for uploaded files
        if (isUploaded) {
            loraEntry.classList.add('uploaded-lora');
        }
        
        loraEntry.innerHTML = `
            <span class="drag-handle" title="Drag to reorder"><i class="fas fa-grip-vertical"></i></span>
            <input type="text" placeholder="username/model-name or /path/to/lora.safetensors" class="lora-name" ${isUploaded ? 'readonly' : ''}>
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

    async removeLoraEntry(loraEntry) {
        const index = this.loraEntries.indexOf(loraEntry);
        if (index > -1) {
            this.loraEntries.splice(index, 1);
        }
        
        // If this is an uploaded LoRA, delete it from the server
        if (loraEntry.classList.contains('uploaded-lora')) {
            const nameInput = loraEntry.querySelector('.lora-name');
            if (nameInput) {
                const filename = nameInput.value;
                try {
                    // Find the stored name from the available LoRAs
                    const response = await fetch(`${this.hostBase}/loras`);
                    if (response.ok) {
                        const data = await response.json();
                        const uploadedLora = data.uploaded?.find(lora => 
                            lora.original_name === filename || lora.stored_name === filename
                        );
                        
                        if (uploadedLora) {
                            // Delete from server
                            const deleteResponse = await fetch(`${this.hostBase}/remove-lora/${uploadedLora.stored_name}`, {
                                method: 'DELETE'
                            });
                            
                            if (!deleteResponse.ok) {
                                console.warn('Failed to delete LoRA from server');
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error deleting LoRA from server:', error);
                }
            }
        }
        
        loraEntry.remove();
        // Ensure the array matches DOM after removal
        const loraList = document.getElementById('lora-list');
        this.loraEntries = Array.from(loraList.children);
        
        // Update Add LoRA button state after removal
        this.updateAddLoraButtonState();
    }

    // Load available LoRAs from server (new design)
    async loadAvailableLoras() {
        // Reset
        this.availableLoras = [];

        // Default LoRA
        this.availableLoras.push({
            name: '/data/weights/lora_checkpoints/Studio_Ghibli_Flux.safetensors',
            weight: 1.0,
            type: 'default',
            displayName: '21j3h123/realEarthKontext/lora_emoji.safetensors (Default)'
        });

        try {
            const resp = await fetch(`${this.hostBase}/loras`);
            if (resp.ok) {
                const data = await resp.json();
                if (data.uploaded && Array.isArray(data.uploaded)) {
                    data.uploaded.forEach(item => {
                        this.availableLoras.push({
                            name: item.original_name || item.stored_name,
                            weight: 1.0,
                            type: 'uploaded',
                            storedName: item.stored_name,
                            displayName: `${item.original_name || item.stored_name} (Uploaded)`,
                            size: item.size,
                            timestamp: item.timestamp
                        });
                    });
                }
            }
        } catch (e) {
            console.warn('Failed to load server LoRAs:', e);
        }

        // Dedupe and populate dropdown
        this.removeDuplicateLoras();
        this.populateLoraDropdown();
    }

    // Dedupe available loras
    removeDuplicateLoras() {
        const seen = new Set();
        this.availableLoras = this.availableLoras.filter(lora => {
            const key = lora.storedName || lora.name;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    // Populate dropdown (#lora-dropdown)
    populateLoraDropdown() {
        const dropdown = document.getElementById('lora-dropdown');
        if (!dropdown) return;

        while (dropdown.options.length > 1) {
            dropdown.remove(1);
        }

        this.availableLoras.forEach(lora => {
            const option = document.createElement('option');
            option.value = lora.name;
            option.textContent = lora.displayName || lora.name;
            option.dataset.loraData = JSON.stringify(lora);
            dropdown.appendChild(option);
        });
    }

    // Render applied LoRAs list
    renderAppliedLoras() {
        const container = document.getElementById('applied-lora-list');
        if (!container) return;
        container.innerHTML = '';

        this.appliedLoras.forEach((lora, index) => {
            const item = document.createElement('div');
            item.className = 'lora-item applied-lora-item';
            item.innerHTML = `
                <div class="lora-info">
                    <div class="lora-name-container">
                        <span class="lora-name" title="${lora.name}">${lora.name}</span>
                    </div>
                    <div class="lora-meta">
                        ${lora.size ? `<span class="lora-size">${this.formatFileSize ? this.formatFileSize(lora.size) : ''}</span>` : ''}
                        ${lora.timestamp ? `<span class="lora-date">${this.formatDate ? this.formatDate(lora.timestamp) : ''}</span>` : ''}
                    </div>
                </div>
                <div class="weight-control">
                    <input type="number" class="weight-input" value="${lora.weight}" min="0" max="2" step="0.1" data-index="${index}">
                </div>
                <button class="btn btn-sm btn-danger remove-from-applied" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;

            const weightInput = item.querySelector('.weight-input');
            const removeBtn = item.querySelector('.remove-from-applied');
            weightInput.addEventListener('input', (e) => {
                const newWeight = parseFloat(e.target.value);
                if (!isNaN(newWeight)) {
                    this.appliedLoras[index].weight = newWeight;
                    this.updateApiCommand();
                }
            });
            removeBtn.addEventListener('click', () => this.removeFromApplied(index));

            container.appendChild(item);
        });
    }

    // Add selected from dropdown to applied
    addSelectedLora() {
        const dropdown = document.getElementById('lora-dropdown');
        if (!dropdown || !dropdown.value) {
            this.showError('Please select a LoRA first');
            return;
        }
        const selectedOption = dropdown.options[dropdown.selectedIndex];
        const loraData = JSON.parse(selectedOption.dataset.loraData);

        const exists = this.appliedLoras.find(item => item.name === loraData.name);
        if (exists) {
            this.showError('This LoRA is already applied');
            return;
        }

        this.appliedLoras.push({
            name: loraData.name,
            weight: loraData.weight,
            type: loraData.type,
            storedName: loraData.storedName,
            size: loraData.size,
            timestamp: loraData.timestamp
        });

        this.renderAppliedLoras();
        this.updateApiCommand();
        dropdown.value = '';
        this.showSuccess(`LoRA "${loraData.name}" added to applied list`);
    }

    // Remove applied by index
    removeFromApplied(index) {
        const removed = this.appliedLoras[index];
        this.appliedLoras.splice(index, 1);
        this.renderAppliedLoras();
        this.updateApiCommand();
        this.showSuccess(`LoRA "${removed?.name || ''}" removed from applied list`);
    }

    // Add default LoRA into applied if desired
    addDefaultLora() {
        const defaultLora = this.availableLoras.find(l => l.type === 'default');
        if (defaultLora) {
            this.appliedLoras.push({
                name: defaultLora.name,
                weight: defaultLora.weight,
                type: defaultLora.type,
                storedName: defaultLora.storedName,
                size: defaultLora.size,
                timestamp: defaultLora.timestamp
            });
            this.renderAppliedLoras();
        }
    }

    // Clear all applied
    clearAllLoras() {
        this.appliedLoras = [];
        this.renderAppliedLoras();
        this.updateApiCommand();
        this.showSuccess('All LoRAs cleared');
    }

    // (No-op now; selection handled in addSelectedLora)
    onLoraSelectionChange() {}

    // Get currently selected LoRA info (not used in new design)
    getSelectedLoraInfo() { return null; }

    getLoraConfigs() {
        // Serialize applied LoRAs for API
        return this.appliedLoras.map(lora => ({
            name: lora.storedName || lora.name,
            weight: lora.weight,
            isUploaded: lora.type === 'uploaded'
        }));
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
            height: parseInt(document.getElementById('height').value),
            guidance_scale: parseFloat(document.getElementById('guidance_scale').value)
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

        // Add upscaler parameters
        const upscaleCheckbox = document.getElementById('upscale');
        if (upscaleCheckbox && upscaleCheckbox.checked) {
            params.upscale = true;
            const upscaleFactor = document.getElementById('upscale-factor');
            if (upscaleFactor) {
                params.upscale_factor = parseInt(upscaleFactor.value);
            }
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
                            <span>${result.width || params.width}×${result.height || params.height}</span>
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

    // Removed BF16 comparison UI; only single-model FP4 flow remains



    showImageModal(result, params, imageUrl) {
        const modal = document.getElementById('image-modal');
        const modalImage = document.getElementById('modal-image');
        const modalPrompt = document.getElementById('modal-prompt');
        const modalParams = document.getElementById('modal-params');
        const modalTime = document.getElementById('modal-time');

        modalImage.src = imageUrl;
        modalPrompt.textContent = params.prompt;
        modalParams.textContent = `${result.width || params.width}×${result.height || params.height}`;
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
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        console.log('✅ Success:', message);
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds for success, 10 seconds for errors
        const timeout = type === 'error' ? 10000 : 5000;
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, timeout);

        // Add slide-in animation
        setTimeout(() => {
            notification.classList.add('notification-show');
        }, 100);
    }

    showUploadProgress(filename) {
        // Show the upload progress container
        const progressContainer = document.getElementById('upload-progress-container');
        const filenameElement = document.getElementById('upload-filename');
        const statusElement = document.getElementById('upload-status');
        const progressFill = document.getElementById('upload-progress-fill');
        
        if (progressContainer && filenameElement && statusElement && progressFill) {
            filenameElement.textContent = filename;
            statusElement.textContent = '0%';
            progressFill.style.width = '0%';
            progressContainer.classList.remove('hidden');
            
            // Store references for progress updates
            this.uploadProgressContainer = progressContainer;
            this.uploadProgressFill = progressFill;
        }
    }

    updateUploadProgress(percent) {
        if (this.uploadProgressFill) {
            this.uploadProgressFill.style.width = percent + '%';
            
            // Also update the status text with percentage
            const statusElement = document.getElementById('upload-status');
            if (statusElement) {
                statusElement.textContent = `${Math.round(percent)}%`;
            }
        }
    }

    hideUploadProgress() {
        const progressContainer = document.getElementById('upload-progress-container');
        if (progressContainer) {
            progressContainer.classList.add('hidden');
        }

        // Clear references
        this.uploadProgressContainer = null;
        this.uploadProgressFill = null;
    }

    triggerFileUpload() {
        // Trigger the hidden file input
        document.getElementById('lora-file-input').click();
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Check file type
        const allowedTypes = ['.safetensors', '.bin', '.pt', '.pth'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            this.showError(`Invalid file type. Please upload a LoRA file (${allowedTypes.join(', ')})`);
            return;
        }

        // Check file size (max 1GB)
        const maxSize = 1024 * 1024 * 1024; // 1GB
        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 1GB.');
            return;
        }

        try {
            // Show upload progress indicator
            this.showUploadProgress(file.name);
            
            // Upload the file using XMLHttpRequest for progress tracking
            const uploadResult = await this.uploadFileWithProgress(file);
            
            // Show completion briefly
            this.updateUploadProgress(100);
            const statusElement = this.uploadProgressContainer?.querySelector('.upload-status');
            if (statusElement) {
                statusElement.textContent = 'Complete!';
            }
            
            // Wait a moment to show completion, then hide
            setTimeout(() => {
                this.hideUploadProgress();
                
                // Refresh the LoRA list to show the newly uploaded file
                this.loadAvailableLoras();
                
                this.showSuccess(`LoRA file "${file.name}" uploaded successfully!`);
                
                // Reset the file input
                event.target.value = '';
            }, 500);
            
        } catch (error) {
            // Hide upload progress on error
            this.hideUploadProgress();
            this.showError(`Failed to upload file: ${error.message}`);
        }
    }

    uploadFileWithProgress(file) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            // Track upload progress
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    this.updateUploadProgress(percentComplete);
                }
            });
            
            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (e) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            });
            
            // Handle errors
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });
            
            // Handle abort
            xhr.addEventListener('abort', () => {
                reject(new Error('Upload was cancelled'));
            });
            
            // Open and send the request
            xhr.open('POST', `${this.hostBase}/upload-lora`);
            xhr.send(this.createFormData(file));
        });
    }

    createFormData(file) {
        const formData = new FormData();
        formData.append('file', file);
        return formData;
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
