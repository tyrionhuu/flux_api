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
        this.addLoraEntry('21j3h123/realEarthKontext/blob/main/lora_emoji.safetensors', 1.0);
        
        // Ensure DOM is fully loaded before updating button visibility
        setTimeout(() => {
            this.updateButtonVisibility(); // Set initial button state
        }, 500);
    }

    setupEventListeners() {
        // Generate buttons
        const generateBtn = document.getElementById('generate-btn');
        const generateWithImageBtn = document.getElementById('generate-with-image-btn');
        
        if (generateBtn) generateBtn.addEventListener('click', () => this.generateImage());
        if (generateWithImageBtn) generateWithImageBtn.addEventListener('click', () => this.generateImageWithImage());

        // Random seed button
        const randomSeedBtn = document.getElementById('random-seed');
        if (randomSeedBtn) randomSeedBtn.addEventListener('click', () => this.randomSeed());
        
        // LoRA controls
        const addLoraBtnMain = document.getElementById('add-lora');
        if (addLoraBtnMain) addLoraBtnMain.addEventListener('click', () => this.addLoraEntry());
        
        // LoRA file upload
        const uploadLoraBtn = document.getElementById('upload-lora');
        const loraFileInput = document.getElementById('lora-file-input');
        
        if (uploadLoraBtn) uploadLoraBtn.addEventListener('click', () => this.triggerFileUpload());
        if (loraFileInput) loraFileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        
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
        if (loraList) {
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
        }

        // Clear history (optional element)
        const clearHistoryBtn = document.getElementById('clear-history');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }

        // Upscaler checkbox
        const upscaleCheckbox = document.getElementById('upscale');
        if (upscaleCheckbox) {
            upscaleCheckbox.addEventListener('change', (e) => {
                const upscaleFactorContainer = document.getElementById('upscale-factor-container');
                if (upscaleFactorContainer) {
                    upscaleFactorContainer.style.display = e.target.checked ? 'block' : 'none';
                }
                // Update API command when upscaler changes
                this.updateApiCommand();
            });
        }
        
        // Image upload functionality
        this.setupImageUpload();
        
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
        const closeModalBtn = document.getElementById('close-modal');
        const downloadImageBtn = document.getElementById('download-image');
        
        if (closeModalBtn) closeModalBtn.addEventListener('click', () => this.closeModal());
        if (downloadImageBtn) downloadImageBtn.addEventListener('click', () => this.downloadCurrentImage());

        // Close modal on backdrop click
        const imageModal = document.getElementById('image-modal');
        if (imageModal) {
            imageModal.addEventListener('click', (e) => {
                if (e.target.id === 'image-modal') {
                    this.closeModal();
                }
            });
        }
        
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
        
        // Auto-update API command when parameters change
        const promptInput = document.getElementById('prompt');
        const widthSelect = document.getElementById('width');
        const heightSelect = document.getElementById('height');
        const seedInput = document.getElementById('seed');
        
        if (promptInput) promptInput.addEventListener('input', () => this.updateApiCommand());
        if (widthSelect) widthSelect.addEventListener('change', () => this.updateApiCommand());
        if (heightSelect) heightSelect.addEventListener('change', () => this.updateApiCommand());
        if (seedInput) seedInput.addEventListener('input', () => this.updateApiCommand());
        
        // Also update API command when image upload changes
        const imageUpload = document.getElementById('image-upload');
        if (imageUpload) {
            imageUpload.addEventListener('change', () => {
                // Wait a bit for the image preview to update, then update API command
                setTimeout(() => this.updateApiCommand(), 100);
            });
        }
        
        // Update API command when image is removed
        const removeImageBtn = document.getElementById('remove-image');
        if (removeImageBtn) {
            removeImageBtn.addEventListener('click', () => {
                // Wait a bit for the image preview to update, then update API command
                setTimeout(() => this.updateApiCommand(), 100);
            });
        }
        
        // Update API command when image strength/guidance changes
        const imageStrength = document.getElementById('image-strength');
        const imageGuidance = document.getElementById('image-guidance');
        if (imageStrength) imageStrength.addEventListener('input', () => this.updateApiCommand());
        if (imageGuidance) imageGuidance.addEventListener('input', () => this.updateApiCommand());
        
        // Update API command when LoRAs change (add/remove/modify)
        const addLoraBtn = document.getElementById('add-lora');
        if (addLoraBtn) {
            addLoraBtn.addEventListener('click', () => {
                // Wait a bit for the LoRA to be added, then update API command
                setTimeout(() => this.updateApiCommand(), 100);
            });
        }
        
        // Initial API command display
        this.updateApiCommand();
        
        // Copy API command button
        const copyApiCommandBtn = document.getElementById('copy-api-command');
        if (copyApiCommandBtn) {
            console.log('Copy button found, adding event listener');
            copyApiCommandBtn.addEventListener('click', (e) => {
                console.log('Copy button clicked!');
                e.preventDefault();
                this.copyApiCommand();
            });
        } else {
            console.error('Copy button not found!');
        }
        
        // Test if API command element exists
        setTimeout(() => {
            const commandElement = document.getElementById('api-command');
            if (commandElement) {
                console.log('API command element found, content:', commandElement.textContent);
            } else {
                console.error('API command element not found');
            }
        }, 1000);

    }

    randomSeed() {
        const seedInput = document.getElementById('seed');
        seedInput.value = Math.floor(Math.random() * 4294967295);
        
        // Update API command when seed changes
        this.updateApiCommand();
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

        // Add event listeners for name and weight changes to update API command
        nameInput.addEventListener('input', () => this.updateApiCommand());
        weightInput.addEventListener('input', () => this.updateApiCommand());

        // Drag events
        loraEntry.addEventListener('dragstart', () => {
            loraEntry.classList.add('dragging');
        });
        loraEntry.addEventListener('dragend', () => {
            loraEntry.classList.remove('dragging');
            // Re-sync array with DOM order
            this.loraEntries = Array.from(loraList.children);
            // Update API command after reordering
            this.updateApiCommand();
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
        
        // Clean up uploaded file data if this was an uploaded LoRA
        if (loraEntry.classList.contains('uploaded-lora')) {
            const nameInput = loraEntry.querySelector('.lora-name');
            if (nameInput && this.uploadedFiles) {
                this.uploadedFiles.delete(nameInput.value);
            }
        }
        
        loraEntry.remove();
        // Ensure the array matches DOM after removal
        const loraList = document.getElementById('lora-list');
        this.loraEntries = Array.from(loraList.children);
        
        // Update Add LoRA button state after removal
        this.updateAddLoraButtonState();
        
        // Update API command after removal
        this.updateApiCommand();
    }

    getLoraConfigs() {
        const configs = [];
        for (const entry of this.loraEntries) {
            const name = entry.querySelector('.lora-name').value.trim();
            const weight = parseFloat(entry.querySelector('.lora-weight').value);
            if (name && !isNaN(weight)) {
                // Check if this is an uploaded file
                const isUploaded = entry.classList.contains('uploaded-lora');
                configs.push({ 
                    name, 
                    weight, 
                    isUploaded,
                    file: isUploaded && this.uploadedFiles ? this.uploadedFiles.get(name) : null
                });
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
            let response;
            
            if (this.hasUploadedImage()) {
                // Use image upload generation endpoint
                response = await this.generateImageWithUpload();
            } else {
                // Use regular text-to-image generation
                const params = this.getGenerationParams();
                response = await fetch(`${this.hostBase}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
            }

            if (!response.ok) {
                let detail = 'Unknown error';
                try {
                    const error = await response.json();
                    detail = error.detail || JSON.stringify(error);
                } catch (_) {
                    const text = await response.text();
                    detail = text?.slice(0, 500) || detail;
                }
                throw new Error(`Generation failed: ${detail}`);
            }

            // Robust JSON parsing (handle proxies returning empty/HTML)
            let result;
            try {
                result = await response.json();
            } catch (_) {
                const text = await response.text();
                try {
                    result = JSON.parse(text);
                } catch (_) {
                    throw new Error(`Invalid JSON response: ${text?.slice(0, 500) || 'empty body'}`);
                }
            }
            console.log('Generation completed');
            
            // Show single image without model identification
            const params = this.getGenerationParams();
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
    
    async generateImageWithUpload() {
        const formData = new FormData();
        
        if (this.serverUploadedImagePath) {
            formData.append('uploaded_image_path', this.serverUploadedImagePath);
        } else if (this.uploadedImageFile) {
            formData.append('file', this.uploadedImageFile);
        } else {
            throw new Error('No image selected or uploaded');
        }
        
        // Add text prompt
        formData.append('prompt', document.getElementById('prompt').value.trim());
        
        // Add LoRA configurations
        const loraConfigs = this.getLoraConfigs();
        if (loraConfigs.length > 0) {
            formData.append('loras', JSON.stringify(loraConfigs));
        }
        
        // Add other parameters
        formData.append('width', document.getElementById('width').value);
        formData.append('height', document.getElementById('height').value);
        const seed = document.getElementById('seed').value;
        if (seed) formData.append('seed', seed);
        const upscaleCheckbox = document.getElementById('upscale');
        if (upscaleCheckbox && upscaleCheckbox.checked) {
            formData.append('upscale', 'true');
            const upscaleFactor = document.getElementById('upscale-factor');
            if (upscaleFactor) formData.append('upscale_factor', upscaleFactor.value);
        }
        const imageParams = this.getImageUploadParams();
        if (imageParams) {
            // image_strength and image_guidance removed
        }
        
        return fetch(`${this.hostBase}/upload-image-generate`, { method: 'POST', body: formData });
    }

    async generateImageWithImage() {
        if (this.isGenerating) return;

        const prompt = document.getElementById('prompt').value.trim();
        if (!prompt) {
            this.showError('Please enter a prompt');
            return;
        }

        if (!this.hasUploadedImage()) {
            this.showError('Please upload an image first');
            return;
        }

        this.isGenerating = true;
        this.showGenerationStatus(true);
        this.updateGenerateButton(true);

        try {
            const formData = new FormData();
            
            // Always send direct image-to-image payload to the supported endpoint
            const endpoint = '/generate-with-image';
            formData.append('image', this.uploadedImageFile);
            
            formData.append('prompt', prompt);
            
            // Add width and height parameters
            const width = document.getElementById('width').value;
            const height = document.getElementById('height').value;
            formData.append('width', width);
            formData.append('height', height);
            
            // Add LoRA configurations
            const loraConfigs = this.getLoraConfigs();
            if (loraConfigs.length > 0) {
                formData.append('loras', JSON.stringify(loraConfigs));
            }
            
            // Add other parameters
            const seed = document.getElementById('seed').value;
            if (seed) formData.append('seed', seed);
            
            const response = await fetch(`${this.hostBase}${endpoint}`, { 
                method: 'POST', 
                body: formData 
            });

            if (!response.ok) {
                let detail = 'Unknown error';
                try {
                    const error = await response.json();
                    detail = error.detail || JSON.stringify(error);
                } catch (_) {
                    const text = await response.text();
                    detail = text?.slice(0, 500) || detail;
                }
                throw new Error(`Generation failed: ${detail}`);
            }

            const result = await response.json();
            console.log('Image-to-image generation completed');
            
            // Show single image
            const params = this.getGenerationParams();
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
    

    setupImageUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('image-upload');
        const removeBtn = document.getElementById('remove-image');
        const uploadStatusLabel = document.getElementById('upload-status-label');
        
        if (!uploadArea || !fileInput || !removeBtn) {
            console.error('Image upload elements not found');
            return;
        }
        
        // Click on area to open file dialog
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // File selection
        fileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Remove image
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeUploadedImage();
            this.serverUploadedImagePath = null;
            if (uploadStatusLabel) uploadStatusLabel.style.display = 'none';
            // Update API command when image is removed
            this.updateApiCommand();
        });
        

        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.handleImageUpload({ target: fileInput });
            }
        });
        
        // Setup slider event listeners
        this.setupImageUploadSliders();
        
        // Add event listener for upscale factor changes
        const upscaleFactorSelect = document.getElementById('upscale-factor');
        if (upscaleFactorSelect) {
            upscaleFactorSelect.addEventListener('change', () => this.updateApiCommand());
        }
    }
    
    setupImageUploadSliders() {
        const imageStrengthSlider = document.getElementById('image-strength');
        const imageStrengthValue = document.getElementById('image-strength-value');
        const imageGuidanceSlider = document.getElementById('image-guidance');
        const imageGuidanceValue = document.getElementById('image-guidance-value');
        
        if (imageStrengthSlider && imageStrengthValue) {
            imageStrengthValue.textContent = imageStrengthSlider.value;
            imageStrengthSlider.addEventListener('input', (e) => {
                imageStrengthValue.textContent = e.target.value;
                // Update API command when image strength changes
                this.updateApiCommand();
            });
        }
        
        if (imageGuidanceSlider && imageGuidanceValue) {
            imageGuidanceValue.textContent = imageGuidanceSlider.value;
            imageGuidanceSlider.addEventListener('input', (e) => {
                imageGuidanceValue.textContent = e.target.value;
                // Update API command when image guidance changes
                this.updateApiCommand();
            });
        }
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select an image file');
            return;
        }
        
        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('Image file too large (max 10MB)');
            return;
        }
        
        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.showImagePreview(e.target.result, file.name);
        };
        reader.readAsDataURL(file);
        
        // Store file for later use
        this.uploadedImageFile = file;
        
        // Try to get the full file path (browsers may not expose this for security)
        // We'll show a helpful message about using the full path
        this.uploadedImagePath = file.name;
        
        // Update button visibility
        this.updateButtonVisibility();
        
        // Auto-upload to server after selection
        this.autoUploadSelectedImage();
        
        // Update API command to show image generation command
        this.updateApiCommand();
    }
    
    async autoUploadSelectedImage() {
        const uploadStatusLabel = document.getElementById('upload-status-label');
        if (!this.uploadedImageFile) return;
        try {
            if (uploadStatusLabel) {
                uploadStatusLabel.style.display = 'inline';
                uploadStatusLabel.textContent = 'Uploading...';
            }
            const formData = new FormData();
            formData.append('file', this.uploadedImageFile);
            const resp = await fetch(`${this.hostBase}/upload-image`, { method: 'POST', body: formData });
            
            console.log('Upload response status:', resp.status);
            console.log('Upload response headers:', Object.fromEntries(resp.headers.entries()));
            
            if (!resp.ok) {
                const responseText = await resp.text();
                console.error('Upload failed response text:', responseText);
                let errorDetail = 'Upload failed';
                try {
                    const err = JSON.parse(responseText);
                    errorDetail = err.detail || errorDetail;
                } catch (parseError) {
                    console.error('Failed to parse error response:', parseError);
                    errorDetail = responseText || `HTTP ${resp.status}`;
                }
                throw new Error(errorDetail);
            }
            
            const responseText = await resp.text();
            console.log('Upload success response text:', responseText);
            
            let json;
            try {
                json = JSON.parse(responseText);
            } catch (parseError) {
                console.error('Failed to parse success response:', parseError);
                console.error('Response text was:', responseText);
                throw new Error('Server returned invalid JSON response');
            }
            
            this.serverUploadedImagePath = json.file_path;
            if (uploadStatusLabel) uploadStatusLabel.textContent = 'Uploaded';
            this.showSuccess('Image uploaded to server');
            
            // Update API command after successful upload
            this.updateApiCommand();
        } catch (err) {
            console.error('Upload error:', err);
            if (uploadStatusLabel) uploadStatusLabel.textContent = 'Upload failed';
            this.showError(err.message || 'Upload failed');
        }
    }
    
    showImagePreview(imageDataUrl, fileName) {
        const uploadArea = document.getElementById('upload-area');
        const uploadPlaceholder = document.getElementById('upload-placeholder');
        const imagePreview = document.getElementById('uploaded-image-preview');
        const uploadControls = document.getElementById('upload-controls');
        
        if (uploadPlaceholder) uploadPlaceholder.classList.add('hidden');
        if (imagePreview) {
            imagePreview.src = imageDataUrl;
            imagePreview.classList.remove('hidden');
        }
        // Show only remove button since upload is automatic
        if (uploadControls) uploadControls.style.display = 'block';
        
        // Update upload area text
        if (uploadArea) {
            uploadArea.title = `Uploaded: ${fileName}`;
        }
    }
    
    removeUploadedImage() {
        const uploadArea = document.getElementById('upload-area');
        const uploadPlaceholder = document.getElementById('upload-placeholder');
        const imagePreview = document.getElementById('uploaded-image-preview');
        const uploadControls = document.getElementById('upload-controls');
        const fileInput = document.getElementById('image-upload');
        
        // Clear file input
        if (fileInput) fileInput.value = '';
        
        // Hide preview and show placeholder
        if (uploadPlaceholder) uploadPlaceholder.classList.remove('hidden');
        if (imagePreview) imagePreview.classList.add('hidden');
        if (uploadControls) uploadControls.style.display = 'none';
        
        this.uploadedImageFile = null;
        
        // Reset upload area
        if (uploadArea) {
            uploadArea.title = '';
        }
        
        // Update button visibility
        this.updateButtonVisibility();
        
        // Update API command when image is removed
        this.updateApiCommand();
    }
    
    hasUploadedImage() {
        return this.uploadedImageFile !== null && this.uploadedImageFile !== undefined;
    }

    updateButtonVisibility() {
        const generateBtn = document.getElementById('generate-btn');
        const generateWithImageBtn = document.getElementById('generate-with-image-btn');
        
        if (this.hasUploadedImage()) {
            generateBtn.style.display = 'none';
            generateWithImageBtn.style.display = 'block';
        } else {
            generateBtn.style.display = 'block';
            generateWithImageBtn.style.display = 'none';
        }
    }
    
    getImageUploadParams() {
        if (!this.hasUploadedImage()) return null;
        
        return {
            // image_strength and image_guidance removed
        };
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
        // Build the JSON payload first, then escape it properly for shell
        const jsonPayload = {
            prompt: prompt,
            width: parseInt(width),
            height: parseInt(height)
        };
        
        if (seed) {
            jsonPayload.seed = parseInt(seed);
        }
        
        if (loras && loras.length > 0) {
            jsonPayload.loras = loras.map(lora => ({
                name: lora.name,
                weight: parseFloat(lora.weight)
            }));
        }
        
        // Escape the JSON string for shell
        const escapedJson = JSON.stringify(jsonPayload).replace(/"/g, '\\"');
        const command = `curl -s -X POST "${window.location.origin}/generate-and-return-image" -H "Content-Type: application/json" -d "${escapedJson}" -o "generated_image.png"`;
        
        commandElement.textContent = command;
        commandSection.classList.remove('hidden');
    }

    updateApiCommand() {
        const commandElement = document.getElementById('api-command');
        const loadingElement = document.getElementById('api-command-loading');
        if (!commandElement) return;
        
        // Show loading state briefly
        if (loadingElement) {
            loadingElement.style.display = 'flex';
            commandElement.style.display = 'none';
        }
        
        // Get current LoRA configuration
        const loras = this.getLoraConfigs();
        const prompt = document.getElementById('prompt').value;
        const width = document.getElementById('width').value;
        const height = document.getElementById('height').value;
        const seed = document.getElementById('seed').value;
        
        // Check if there's an uploaded image
        const uploadedImage = document.getElementById('uploaded-image-preview');
        const hasImage = uploadedImage && !uploadedImage.classList.contains('hidden');
        
        let command;
        
        if (hasImage) {
            // Build command for generate-with-image-and-return endpoint
            // Show the filename but remind user to use full local path
            const imageFileName = this.uploadedImageFile ? this.uploadedImageFile.name : 'your_image.jpg';
            command = `# Replace 'your_image.jpg' with the full path to your image file\n`;
            command += `curl -s -X POST "${window.location.origin}/generate-with-image-and-return" -F "image=@/full/path/to/${imageFileName}" -F "prompt=${this.escapeForShell(prompt)}" -F "width=${width}" -F "height=${height}"`;
            
            // Add image strength and guidance if available
            const imageStrength = document.getElementById('image-strength');
            const imageGuidance = document.getElementById('image-guidance');
            // image_strength and image_guidance removed from curl
            
            if (seed) {
                command += ` -F "seed=${seed}"`;
            }
            
            if (loras && loras.length > 0) {
                // Convert LoRAs to JSON string for form data
                const lorasJson = JSON.stringify(loras.map(lora => ({
                    name: lora.name,
                    weight: parseFloat(lora.weight)
                })));
                command += ` -F "loras=${this.escapeForShell(lorasJson)}"`;
            }
            
            // Direct output to file - no need for jq or second curl
            command += ` -o "generated_image.png"`;
        } else {
            // Build command for generate-and-return-image-simple endpoint
            // Build the JSON payload first, then escape it properly for shell
            const jsonPayload = {
                prompt: prompt,
                width: parseInt(width),
                height: parseInt(height)
            };
            
            if (seed) {
                jsonPayload.seed = parseInt(seed);
            }
            
            if (loras && loras.length > 0) {
                jsonPayload.loras = loras.map(lora => ({
                    name: lora.name,
                    weight: parseFloat(lora.weight)
                }));
            }
            
            // Escape the JSON string for shell
            const escapedJson = JSON.stringify(jsonPayload).replace(/"/g, '\\"');
            command = `curl -s -X POST "${window.location.origin}/generate-and-return-image" -H "Content-Type: application/json" -d "${escapedJson}" -o "generated_image.png"`;
        }
        
        commandElement.textContent = command;
        
        // Ensure the API command section is visible
        const commandSection = document.getElementById('api-command-section');
        if (commandSection) {
            commandSection.classList.remove('hidden');
        }
        
        // Show/hide the help message for image-to-image commands
        const helpElement = document.getElementById('api-command-help');
        if (helpElement) {
            if (hasImage) {
                helpElement.style.display = 'block';
            } else {
                helpElement.style.display = 'none';
            }
        }
        
        // Hide loading state and show command after a brief delay
        setTimeout(() => {
            if (loadingElement) {
                loadingElement.style.display = 'none';
            }
            if (commandElement) {
                commandElement.style.display = 'block';
            }
        }, 100);
    }
    
    escapeForShell(text) {
        if (!text) return '';
        // Escape special characters for shell form data
        return text.replace(/"/g, '\\"').replace(/'/g, "\\'").replace(/\$/g, '\\$');
    }
    
    copyApiCommand() {
        console.log('Copy API command function called');
        
        const commandElement = document.getElementById('api-command');
        if (!commandElement) {
            console.error('API command element not found');
            this.showError('API command element not found');
            return;
        }
        
        if (!commandElement.textContent || commandElement.textContent.trim() === '') {
            console.error('No API command text to copy');
            this.showError('No API command to copy');
            return;
        }
        
        const textToCopy = commandElement.textContent.trim();
        console.log('Text to copy:', textToCopy);
        
        // Try modern clipboard API first
        if (navigator.clipboard && navigator.clipboard.writeText) {
            console.log('Using modern clipboard API');
            navigator.clipboard.writeText(textToCopy)
                .then(() => {
                    console.log('Successfully copied to clipboard');
                    this.showSuccess('API command copied to clipboard!');
                })
                .catch((err) => {
                    console.error('Modern clipboard API failed:', err);
                    // Fallback to execCommand
                    this.fallbackCopyToClipboard(textToCopy);
                });
        } else {
            console.log('Modern clipboard API not available, using fallback');
            this.fallbackCopyToClipboard(textToCopy);
        }
    }
    
    fallbackCopyToClipboard(text) {
        try {
            console.log('Using fallback copy method');
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (successful) {
                console.log('Fallback copy successful');
                this.showSuccess('API command copied to clipboard!');
            } else {
                console.error('Fallback copy failed');
                this.showError('Failed to copy API command (fallback method failed)');
            }
        } catch (err) {
            console.error('Fallback copy error:', err);
            this.showError('Failed to copy API command: ' + err.message);
        }
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
        this.showNotification(message, 'success');
    }

    showError(message) {
        console.error('❌ Error:', message);
        this.showNotification(message, 'error');
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
        // Create minimal upload progress indicator
        const progressContainer = document.createElement('div');
        progressContainer.id = 'upload-progress';
        progressContainer.className = 'upload-progress';
        progressContainer.innerHTML = `
            <div class="upload-progress-content">
                <div class="upload-progress-text">
                    <div class="upload-filename">${filename}</div>
                    <div class="upload-status">Uploading...</div>
                </div>
                <div class="upload-progress-bar">
                    <div class="upload-progress-fill"></div>
                </div>
            </div>
        `;

        // Add to page
        document.body.appendChild(progressContainer);

        // Add slide-in animation
        setTimeout(() => {
            progressContainer.classList.add('upload-progress-show');
        }, 100);

        // Store container reference for progress updates
        this.uploadProgressContainer = progressContainer;
        this.uploadProgressFill = progressContainer.querySelector('.upload-progress-fill');
    }

    updateUploadProgress(percent) {
        if (this.uploadProgressFill) {
            this.uploadProgressFill.style.width = percent + '%';
            
            // Also update the status text with percentage
            const statusElement = this.uploadProgressContainer?.querySelector('.upload-status');
            if (statusElement) {
                statusElement.textContent = `Uploading... ${Math.round(percent)}%`;
            }
        }
    }

    hideUploadProgress() {
        const progressContainer = document.getElementById('upload-progress');
        if (progressContainer) {
            // Show completion briefly
            progressContainer.classList.add('upload-progress-complete');
            
            // Remove after animation
            setTimeout(() => {
                if (progressContainer.parentElement) {
                    progressContainer.remove();
                }
            }, 500);
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
                
                // Add the uploaded file to the LoRA list with the server filename
                this.addLoraEntry(uploadResult.filename, 1.0, true); // true indicates it's an uploaded file
                
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