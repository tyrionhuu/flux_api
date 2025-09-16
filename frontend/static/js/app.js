// FLUX API ComfyUI Frontend
class FluxAPI {
    constructor() {
        // Use current origin (protocol + host + port) so UI works on any served port
        this.hostBase = window.location.origin;
        this.isGenerating = false;
        this.appliedLoras = []; // 已应用的LoRA列表
        this.availableLoras = []; // 可选择的LoRA列表
        this.uploadedFiles = new Map(); // 存储上传的文件映射
        
        // LoRA Fusion State Management
        this.isFused = false;
        this.fusedLoras = [];
        this.fusedTimestamp = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableLoras();
        // Removed automatic default LoRA addition
        this.updateApiCommand();

        // Check fusion status on startup
        this.checkFusionStatus();

        setTimeout(() => {
            this.updateButtonVisibility();
        }, 500);
    }

    // Utility functions to reduce code duplication
    getElement(id) {
        return document.getElementById(id);
    }

    getFormValues() {
        return {
            prompt: this.getElement('prompt').value.trim(),
            negativePrompt: (this.getElement('negative_prompt')?.value || '').trim(),
            seed: this.getElement('seed').value,
            numInferenceSteps: this.getElement('inference_steps').value,
            guidanceScale: this.getElement('guidance_scale').value,
            trueCfgScale: this.getElement('true_cfg_scale')?.value || this.getElement('guidance_scale').value
        };
    }

    buildFormData(params = {}) {
        const formData = new FormData();
        const values = this.getFormValues();

        // Common parameters
        formData.append('prompt', params.prompt || values.prompt);
        // width/height are auto-computed on the backend

        // Inference parameters
        if (params.numInferenceSteps || values.numInferenceSteps) {
            formData.append('num_inference_steps', params.numInferenceSteps || values.numInferenceSteps);
        }
        if (params.guidanceScale || values.guidanceScale) {
            formData.append('guidance_scale', params.guidanceScale || values.guidanceScale);
        }

        // Optional parameters
        if (params.seed || values.seed) {
            formData.append('seed', params.seed || values.seed);
        }

        // Negative prompt for form-data endpoints
        if (params.negativePrompt || values.negativePrompt) {
            formData.append('negative_prompt', params.negativePrompt || values.negativePrompt);
        }

        // LoRA configurations
        const loraConfigs = this.getLoraConfigs();
        if (loraConfigs.length > 0) {
            formData.append('loras', JSON.stringify(loraConfigs));
        }

        // Upscale settings
        const upscaleCheckbox = this.getElement('upscale');
        if (upscaleCheckbox && upscaleCheckbox.checked) {
            formData.append('upscale', 'true');
            const upscaleFactor = this.getElement('upscale-factor');
            if (upscaleFactor) formData.append('upscale_factor', upscaleFactor.value);
        }

        // Background removal
        const removeBgCheckbox = this.getElement('remove-background');
        if (removeBgCheckbox && removeBgCheckbox.checked) {
            formData.append('remove_background', 'true');
            const bgStrength = this.getElement('bg_strength');
            if (bgStrength && bgStrength.value) {
                formData.append('bg_strength', bgStrength.value);
            }
        }

        // Downscale settings
        const downscaleCheckbox = this.getElement('downscale');
        if (downscaleCheckbox) {
            const downscaleValue = downscaleCheckbox.checked ? 'true' : 'false';
            console.log(`Downscale checkbox checked: ${downscaleCheckbox.checked}, sending: ${downscaleValue}`);
            formData.append('downscale', downscaleValue);
        }

        return formData;
    }

    async handleFetchResponse(response) {
        if (!response.ok) {
            let detail = `Server error (${response.status}: ${response.statusText})`;
            
            try {
                // Check content type before attempting to parse
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const error = await response.json();
                    detail = error.detail || error.message || JSON.stringify(error);
                } else {
                    // Try to get text content for non-JSON responses
                    const bodyText = await response.text();
                    if (bodyText && bodyText.trim()) {
                        detail = bodyText.slice(0, 500) + (bodyText.length > 500 ? '...' : '');
                    }
                }
            } catch (parseError) {
                console.warn('Failed to parse error response:', parseError);
                // Keep the default error message
            }
            
            throw new Error(`Request failed: ${detail}`);
        }
    
        // Handle successful responses
        try {
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                // For non-JSON responses, return text
                const responseText = await response.text();
                console.warn('Non-JSON response received:', responseText.substring(0, 100));
                return { message: 'Success', data: responseText };
            }
        } catch (parseError) {
            console.error('Response parsing error:', parseError);
            
            try {
                const responseText = await response.text();
                console.error('Response text:', responseText.substring(0, 500));
                throw new Error(`Invalid response format: ${responseText?.slice(0, 200) || 'empty body'}...`);
            } catch (textError) {
                console.error('Failed to read response text:', textError);
                throw new Error(`Invalid response: Unable to read response body`);
            }
        }
    }

    buildJsonPayload(includeInferenceParams = false) {
        const values = this.getFormValues();
        const loras = this.getLoraConfigs();
        const upscaleCheckbox = this.getElement('upscale');
        const removeBgCheckbox = this.getElement('remove-background');

        const jsonPayload = {
            prompt: values.prompt
        };

        if (values.negativePrompt) {
            jsonPayload.negative_prompt = values.negativePrompt;
        }

        if (includeInferenceParams) {
            jsonPayload.num_inference_steps = parseInt(values.numInferenceSteps);
            jsonPayload.guidance_scale = parseFloat(values.guidanceScale);
            if (values.negativePrompt) {
                jsonPayload.true_cfg_scale = parseFloat(values.trueCfgScale);
            }
        }

        if (values.seed) {
            jsonPayload.seed = parseInt(values.seed);
        }

        if (loras && loras.length > 0) {
            jsonPayload.loras = loras.map(lora => ({
                name: lora.name,
                weight: parseFloat(lora.weight)
            }));
        }

        if (upscaleCheckbox && upscaleCheckbox.checked) {
            jsonPayload.upscale = true;
            const upscaleFactor = this.getElement('upscale-factor');
            if (upscaleFactor) {
                jsonPayload.upscale_factor = parseInt(upscaleFactor.value);
            }
        }

        if (removeBgCheckbox && removeBgCheckbox.checked) {
            jsonPayload.remove_background = true;
            const bgStrength = this.getElement('bg_strength');
            if (bgStrength && bgStrength.value) {
                jsonPayload.bg_strength = parseFloat(bgStrength.value);
            }
        }

        // Downscale settings
        const downscaleCheckbox = this.getElement('downscale');
        if (downscaleCheckbox) {
            jsonPayload.downscale = downscaleCheckbox.checked;
        }

        return jsonPayload;
    }

    setupEventListeners() {
        // Generate buttons
        const generateBtn = this.getElement('generate-btn');
        const generateWithImageBtn = this.getElement('generate-with-image-btn');

        if (generateBtn) generateBtn.addEventListener('click', () => this.generateImage());
        if (generateWithImageBtn) generateWithImageBtn.addEventListener('click', () => this.generateImageWithImage());

        // Random seed button
        const randomSeedBtn = this.getElement('random-seed');
        if (randomSeedBtn) randomSeedBtn.addEventListener('click', () => this.randomSeed());

        // Negative prompt input updates API command
        const negativePromptInput = this.getElement('negative_prompt');
        if (negativePromptInput) negativePromptInput.addEventListener('input', () => this.updateApiCommand());

        // True CFG scale slider
        const trueCfgSlider = this.getElement('true_cfg_scale');
        if (trueCfgSlider) {
            trueCfgSlider.addEventListener('input', (e) => {
                const valueDisplay = this.getElement('true_cfg_scale_value');
                if (valueDisplay) {
                    valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
                }
                this.updateApiCommand();
            });
        }

        // LoRA controls
        const addLoraBtnMain = this.getElement('add-custom-lora');
        if (addLoraBtnMain) addLoraBtnMain.addEventListener('click', () => this.addCustomLora());

        // LoRA file upload
        const uploadLoraBtn = this.getElement('upload-lora');
        const loraFileInput = this.getElement('lora-file-input');

        if (uploadLoraBtn) uploadLoraBtn.addEventListener('click', () => this.triggerFileUpload());
        if (loraFileInput) loraFileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        // 添加选中的LoRA按钮
        const addSelectedBtn = this.getElement('add-selected-lora');
        if (addSelectedBtn) {
            addSelectedBtn.addEventListener('click', () => this.addSelectedLora());
        }

        // 清空所有LoRA按钮
        const clearAllBtn = this.getElement('clear-all-loras');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => this.clearAllLoras());
        }

        // 刷新LoRA列表按钮
        const refreshBtn = this.getElement('refresh-loras');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                console.log('Refreshing LoRA list...');
                this.loadAvailableLoras();
                this.showSuccess('LoRA list refreshed');
            });
        }

        // Apply LoRA button
        const applyLoraBtn = this.getElement('apply-lora-btn');
        console.log('Apply LoRA button found:', applyLoraBtn);
        if (applyLoraBtn) {
            applyLoraBtn.addEventListener('click', () => {
                console.log('Apply LoRA button clicked!');
                this.applyLorasPermanently();
            });
        } else {
            console.error('Apply LoRA button not found!');
        }

        // Unfuse LoRAs button
        const unfuseLoraBtn = this.getElement('unfuse-loras-btn');
        if (unfuseLoraBtn) {
            unfuseLoraBtn.addEventListener('click', () => {
                console.log('Unfuse LoRA button clicked!');
                this.unfuseLoras();
            });
        }

        // Upscaler checkbox
        const upscaleCheckbox = this.getElement('upscale');
        if (upscaleCheckbox) {
            upscaleCheckbox.addEventListener('change', (e) => {
                const upscaleFactorContainer = this.getElement('upscale-factor-container');
                if (upscaleFactorContainer) {
                    upscaleFactorContainer.style.display = e.target.checked ? 'block' : 'none';
                }
                // Update API command when upscaler changes
                this.updateApiCommand();
            });
        }

        // Background removal strength wiring
        const removeBgCheckboxInit = this.getElement('remove-background');
        const bgStrengthSlider = this.getElement('bg_strength');
        const bgStrengthValue = this.getElement('bg_strength_value');
        if (removeBgCheckboxInit && bgStrengthSlider) {
            const syncBgStrength = () => {
                bgStrengthSlider.disabled = !removeBgCheckboxInit.checked;
                if (bgStrengthValue) {
                    const v = parseFloat(bgStrengthSlider.value || '0');
                    bgStrengthValue.textContent = v.toFixed(2);
                }
                this.updateApiCommand();
            };
            removeBgCheckboxInit.addEventListener('change', syncBgStrength);
            bgStrengthSlider.addEventListener('input', syncBgStrength);
            syncBgStrength();
        }

        // Inference steps slider
        const inferenceStepsSlider = this.getElement('inference_steps');
        if (inferenceStepsSlider) {
            inferenceStepsSlider.addEventListener('input', (e) => {
                const valueDisplay = this.getElement('inference_steps_value');
                if (valueDisplay) {
                    valueDisplay.textContent = e.target.value;
                }
                // Update API command when inference steps change
                this.updateApiCommand();
            });
        }

        // Guidance scale slider
        const guidanceScaleSlider = this.getElement('guidance_scale');
        if (guidanceScaleSlider) {
            guidanceScaleSlider.addEventListener('input', (e) => {
                const valueDisplay = this.getElement('guidance_scale_value');
                if (valueDisplay) {
                    valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
                }
                // Update API command when guidance scale changes
                this.updateApiCommand();
            });
        }

        // Image upload functionality
        this.setupImageUpload();


        // Modal controls
        const closeModalBtn = this.getElement('close-modal');
        const downloadImageBtn = this.getElement('download-image');

        if (closeModalBtn) closeModalBtn.addEventListener('click', () => this.closeModal());
        if (downloadImageBtn) downloadImageBtn.addEventListener('click', () => this.downloadCurrentImage());

        // Close modal on backdrop click
        const imageModal = this.getElement('image-modal');
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
        const promptInput = this.getElement('prompt');
        const widthSelect = null;
        const heightSelect = null;
        const seedInput = this.getElement('seed');

        if (promptInput) promptInput.addEventListener('input', () => this.updateApiCommand());
        // width/height removed
        if (seedInput) seedInput.addEventListener('input', () => this.updateApiCommand());

        // Also update API command when image upload changes
        const imageUpload = this.getElement('image-upload');
        if (imageUpload) {
            imageUpload.addEventListener('change', () => {
                // Wait a bit for the image preview to update, then update API command
                setTimeout(() => this.updateApiCommand(), 100);
            });
        }

        // Update API command when image is removed
        const removeImageBtn = this.getElement('remove-image');
        if (removeImageBtn) {
            removeImageBtn.addEventListener('click', () => {
                // Wait a bit for the image preview to update, then update API command
                setTimeout(() => this.updateApiCommand(), 100);
            });
        }

        // Update API command when LoRAs change (add/remove/modify)
        const addLoraBtn = this.getElement('add-lora');
        if (addLoraBtn) {
            addLoraBtn.addEventListener('click', () => {
                // Wait a bit for the LoRA to be added, then update API command
                setTimeout(() => this.updateApiCommand(), 100);
            });
        }

        // Initial API command display
        this.updateApiCommand();

        // Copy API command button
        const copyApiCommandBtn = this.getElement('copy-api-command');
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
            const commandElement = this.getElement('api-command');
            if (commandElement) {
                console.log('API command element found, content:', commandElement.textContent);
            } else {
                console.error('API command element not found');
            }
        }, 1000);

    }

    randomSeed() {
        const seedInput = this.getElement('seed');
        seedInput.value = Math.floor(Math.random() * 4294967295);

        // Update API command when seed changes
        this.updateApiCommand();
    }

    // 加载可用的LoRA列表
    async loadAvailableLoras() {
        // 清空现有列表，避免重复
        this.availableLoras = [];

        // No default LoRA added - users must explicitly select LoRAs

        // 从服务器加载上传的LoRA和Hugging Face LoRA
        try {
            const resp = await fetch(`${this.hostBase}/loras`);
            if (resp.ok) {
                const data = await resp.json();
                
                // Add uploaded LoRAs
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
                
                // Add Hugging Face LoRAs
                if (data.huggingface && Array.isArray(data.huggingface)) {
                    data.huggingface.forEach(item => {
                        this.availableLoras.push({
                            name: item.name,
                            weight: 1.0,
                            type: 'huggingface',
                            repoId: item.repo_id,
                            filename: item.filename,
                            displayName: `${item.display_name} (Hugging Face)`,
                            size: item.size,
                            timestamp: item.timestamp
                        });
                    });
                }
            }
        } catch (e) {
            console.warn('Failed to load server LoRAs:', e);
        }

        // Add currently applied LoRAs to available list (so they can be selected again)
        this.appliedLoras.forEach(appliedLora => {
            // Check if this LoRA is already in the available list
            const exists = this.availableLoras.find(lora => lora.name === appliedLora.name);
            if (!exists) {
                this.availableLoras.push({
                    name: appliedLora.name,
                    weight: appliedLora.weight,
                    type: appliedLora.type,
                    storedName: appliedLora.storedName,
                    repoId: appliedLora.repoId,
                    filename: appliedLora.filename,
                    displayName: appliedLora.displayName || `${appliedLora.name} (Applied)`,
                    size: appliedLora.size,
                    timestamp: appliedLora.timestamp
                });
            }
        });

        // 清理重复项
        this.removeDuplicateLoras();
        this.populateLoraDropdown();
    }

    // 移除重复的LoRA条目
    removeDuplicateLoras() {
        const seen = new Set();
        this.availableLoras = this.availableLoras.filter(lora => {
            const key = lora.storedName || lora.name;
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    }

    // 清空LoRA列表
    clearLoraList() {
        this.availableLoras = [];
        this.populateLoraDropdown();
        this.showSuccess('LoRA list cleared');
    }

    // 调试：显示当前LoRA状态
    debugLoraState() {
        console.log('=== LoRA Debug Info ===');
        console.log('Available LoRAs:', this.availableLoras);
        console.log('Applied LoRAs:', this.appliedLoras);
        console.log('Dropdown options:', this.getElement('lora-dropdown')?.options?.length || 'No dropdown found');
        console.log('=======================');
    }

    // 填充LoRA下拉列表
    populateLoraDropdown() {
        const dropdown = this.getElement('lora-dropdown');
        if (!dropdown) return;

        // 清空现有选项（保留第一个提示选项）
        while (dropdown.options.length > 1) {
            dropdown.remove(1);
        }

        // 添加调试信息
        console.log(`Populating LoRA dropdown with ${this.availableLoras.length} items:`, this.availableLoras);

        // 添加所有可用的LoRA
        this.availableLoras.forEach((lora, index) => {
            const option = document.createElement('option');
            option.value = lora.name;
            option.textContent = lora.displayName;
            option.dataset.loraData = JSON.stringify(lora);
            dropdown.appendChild(option);
        });

        console.log(`Dropdown now has ${dropdown.options.length} options`);
    }

    // 渲染已应用的LoRA列表
    renderAppliedLoras() {
        const container = this.getElement('applied-lora-list');
        if (!container) {
            console.error('Applied LoRA list container not found!');
            return;
        }

        console.log('Rendering applied LoRAs:', this.appliedLoras);
        container.innerHTML = '';

        // Show placeholder if no LoRAs are applied
        if (this.appliedLoras.length === 0) {
            container.innerHTML = `
                <div class="lora-empty-placeholder">
                    <i class="fas fa-magic"></i>
                    <p>No LoRAs selected</p>
                    <small>Select LoRAs from the dropdown above to add them here</small>
                </div>
            `;
            return;
        }

        this.appliedLoras.forEach((lora, index) => {
            const item = document.createElement('div');
            item.className = 'lora-item applied-lora-item';
            item.innerHTML = `
                <div class="lora-info">
                    <div class="lora-name-container">
                        <span class="lora-name" title="${lora.name}">${lora.name}</span>
                    </div>
                    <div class="lora-meta">
                        ${lora.size ? `<span class="lora-size">${this.formatFileSize(lora.size)}</span>` : ''}
                        ${lora.timestamp ? `<span class="lora-date">${this.formatDate(lora.timestamp)}</span>` : ''}
                    </div>
                </div>
                                <div class="weight-control">
                    <input type="number" class="weight-input" value="${lora.weight}" min="0" max="2" step="0.1" data-index="${index}">
                </div>
                <button class="btn btn-sm btn-danger remove-from-applied" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;

            // 添加事件监听器
            const weightInput = item.querySelector('.weight-input');
            const removeBtn = item.querySelector('.remove-from-applied');

            // Weight input change
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

    // 添加选中的LoRA到应用列表
    addSelectedLora() {
        const dropdown = this.getElement('lora-dropdown');
        if (!dropdown || !dropdown.value) {
            this.showError('Please select a LoRA first');
            return;
        }

        const selectedOption = dropdown.options[dropdown.selectedIndex];
        console.log('Selected option:', selectedOption);
        console.log('Dataset:', selectedOption.dataset);
        
        const loraData = JSON.parse(selectedOption.dataset.loraData);
        console.log('Parsed LoRA data:', loraData);

        // 检查是否已经在应用列表中
        const exists = this.appliedLoras.find(item => item.name === loraData.name);
        if (exists) {
            this.showError('This LoRA is already applied');
            return;
        }

        // 添加到应用列表
        this.appliedLoras.push({
            name: loraData.name,
            weight: loraData.weight,
            type: loraData.type,
            storedName: loraData.storedName,
            repoId: loraData.repoId,
            filename: loraData.filename,
            size: loraData.size,
            timestamp: loraData.timestamp
        });

        console.log('Applied LoRAs after adding:', this.appliedLoras);
        this.renderAppliedLoras();
        this.updateApiCommand();

        // 重置下拉选择
        dropdown.value = '';

        this.showSuccess(`LoRA "${loraData.name}" added to applied list`);
    }

    // 格式化文件大小
    formatFileSize(bytes) {
        if (!bytes) return '';
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    // 格式化日期
    formatDate(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp * 1000);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    // 从已应用列表移除
    removeFromApplied(index) {
        const removed = this.appliedLoras[index];
        this.appliedLoras.splice(index, 1);
        this.renderAppliedLoras();
        this.updateApiCommand();
        this.showSuccess(`LoRA "${removed.name}" removed from applied list`);
    }

    // addDefaultLora function removed - no default LoRA is applied automatically

    // 清空所有已应用的LoRA
    clearAllLoras() {
        this.appliedLoras = [];
        this.renderAppliedLoras();
        this.updateApiCommand();
        this.showSuccess('All LoRAs cleared');
    }

    // Apply LoRAs permanently (fusion)
    async applyLorasPermanently() {
        if (this.appliedLoras.length === 0) {
            this.showError('No LoRAs to apply. Please add LoRAs first.');
            return;
        }

        try {
            this.showLoading('Fusing LoRAs...');
            
            const response = await fetch(`${this.hostBase}/apply-lora-permanent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    loras: this.appliedLoras.map(lora => ({
                        name: lora.storedName || lora.name,
                        weight: lora.weight
                    }))
                })
            });

            // Handle different types of errors
            if (!response.ok) {
                let errorMessage = `Server error (${response.status}: ${response.statusText})`;
                
                try {
                    // Try to parse error response as JSON
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        const errorData = await response.json();
                        errorMessage = errorData.detail || errorData.message || errorMessage;
                    } else {
                        // If not JSON, try to get text content
                        const errorText = await response.text();
                        if (errorText.trim()) {
                            errorMessage = errorText.substring(0, 200) + (errorText.length > 200 ? '...' : '');
                        }
                    }
                } catch (parseError) {
                    console.warn('Failed to parse error response:', parseError);
                    // Use the default error message
                }
                
                throw new Error(errorMessage);
            }

            // Parse successful response
            let result;
            try {
                result = await response.json();
            } catch (parseError) {
                console.warn('Failed to parse success response as JSON:', parseError);
                result = { message: 'LoRAs fused successfully (server response parsing failed)' };
            }
            
            // Update fusion state
            this.isFused = true;
            this.fusedLoras = [...this.appliedLoras];
            this.fusedTimestamp = Date.now();
            
            this.updateFusedState();
            this.showSuccess(result.message || 'LoRAs fused successfully');
            
            // Update button text
            const applyBtn = document.getElementById('apply-lora-btn');
            if (applyBtn) {
                applyBtn.innerHTML = '<i class="fas fa-link"></i> LoRAs Fused';
                applyBtn.className = 'btn btn-success fused';
            }

        } catch (error) {
            console.error('LoRA fusion error:', error);
            this.showError(error.message || 'Failed to fuse LoRAs');
        } finally {
            this.hideLoading();
        }
    }

    // Unfuse LoRAs
    async unfuseLoras() {
        try {
            this.showLoading('Unfusing LoRAs...');
            
            const response = await fetch(`${this.hostBase}/unfuse-loras`, {
                method: 'DELETE'
            });

            // Handle different types of errors
            if (!response.ok) {
                let errorMessage = `Server error (${response.status}: ${response.statusText})`;
                
                try {
                    // Try to parse error response as JSON
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        const errorData = await response.json();
                        errorMessage = errorData.detail || errorData.message || errorMessage;
                    } else {
                        // If not JSON, try to get text content
                        const errorText = await response.text();
                        if (errorText.trim()) {
                            errorMessage = errorText.substring(0, 200) + (errorText.length > 200 ? '...' : '');
                        }
                    }
                } catch (parseError) {
                    console.warn('Failed to parse error response:', parseError);
                    // Use the default error message
                }
                
                throw new Error(errorMessage);
            }

            // Parse successful response
            let result;
            try {
                result = await response.json();
            } catch (parseError) {
                console.warn('Failed to parse success response as JSON:', parseError);
                result = { message: 'LoRAs unfused successfully (server response parsing failed)' };
            }
            
            // Reset fusion state
            this.isFused = false;
            this.fusedLoras = [];
            this.fusedTimestamp = null;
            
            // Clear applied LoRAs to prevent them from being sent during generation
            this.appliedLoras = [];
            
            // Update the UI to reflect no LoRAs are selected
            this.renderAppliedLoras();
            
            // Also reset the LoRA dropdown selection
            const dropdown = document.getElementById('lora-dropdown');
            if (dropdown) {
                dropdown.value = '';
            }
            
            this.updateFusedState();
            this.showSuccess(result.message || 'LoRAs unfused successfully');
            
            // Update button text
            const applyBtn = document.getElementById('apply-lora-btn');
            if (applyBtn) {
                applyBtn.innerHTML = '<i class="fas fa-magic"></i> Apply LoRAs';
                applyBtn.className = 'btn btn-primary';
            }

        } catch (error) {
            console.error('LoRA unfusion error:', error);
            this.showError(error.message || 'Failed to unfuse LoRAs');
        } finally {
            this.hideLoading();
        }
    }

    // 获取LoRA配置（用于API调用）
    getLoraConfigs() {
        // If LoRAs are fused, return empty array since the fusion is already applied to the model
        if (this.isFused) {
            return [];
        }
        
        return this.appliedLoras.map(lora => ({
            name: lora.storedName || lora.name,
            weight: lora.weight,
            isUploaded: lora.type === 'uploaded',
            isHuggingFace: lora.type === 'huggingface',
            repoId: lora.repoId,
            filename: lora.filename
        }));
    }

    // Update UI to reflect fused state
    updateFusedState() {
        const applyBtn = this.getElement('apply-lora-btn');
        const unfuseBtn = this.getElement('unfuse-loras-btn');
        const fusionStatus = this.getElement('lora-fusion-status');
        
        if (this.isFused) {
            // Show unfuse button and hide apply button
            if (unfuseBtn) unfuseBtn.style.display = 'inline-block';
            if (applyBtn) applyBtn.style.display = 'none';
            
            // Show fused LoRA status
            this.showFusedLoraStatus();
        } else {
            // Show apply button and hide unfuse button
            if (applyBtn) applyBtn.style.display = 'inline-block';
            if (unfuseBtn) unfuseBtn.style.display = 'none';
            
            // Hide fused LoRA status
            this.hideFusedLoraStatus();
        }
    }

    // Show fused LoRA status
    showFusedLoraStatus() {
        const fusionStatus = this.getElement('lora-fusion-status');
        const fusionCount = this.getElement('fusion-count');
        const fusionTimestamp = this.getElement('fusion-timestamp');
        
        if (fusionStatus) {
            fusionStatus.style.display = 'block';
            fusionStatus.classList.add('show');
        }
        
        if (fusionCount && this.fusedLoras.length > 0) {
            fusionCount.textContent = `${this.fusedLoras.length} LoRA${this.fusedLoras.length > 1 ? 's' : ''}`;
        }
        
        if (fusionTimestamp && this.fusedTimestamp) {
            const date = new Date(this.fusedTimestamp);
            fusionTimestamp.textContent = `Fused at ${date.toLocaleTimeString()}`;
        }
    }

    // Hide fused LoRA status
    hideFusedLoraStatus() {
        const fusionStatus = this.getElement('lora-fusion-status');
        if (fusionStatus) {
            fusionStatus.style.display = 'none';
            fusionStatus.classList.remove('show');
        }
    }

    // Check fusion status on startup
    async checkFusionStatus() {
        try {
            const response = await fetch(`${this.hostBase}/fused-lora-status`);
            
            if (response.ok) {
                let status;
                try {
                    status = await response.json();
                } catch (parseError) {
                    console.warn('Failed to parse fusion status response:', parseError);
                    return; // Exit gracefully if we can't parse the response
                }
                
                this.isFused = status.is_fused || false;
                if (status.fused_info) {
                    this.fusedLoras = status.fused_info.fused_lora_configs || [];
                    this.fusedTimestamp = status.fused_info.fused_timestamp ? 
                        status.fused_info.fused_timestamp * 1000 : null; // Convert to milliseconds
                }
                this.updateFusedState();
            } else {
                console.warn(`Failed to check fusion status: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.warn('Error checking fusion status (non-critical):', error);
            // Don't show error to user as this is a background check
        }
    }

    // 添加自定义LoRA
    async addCustomLora() {
        const customName = prompt('Enter LoRA name (Hugging Face repo ID or local path):');
        if (!customName || !customName.trim()) return;

        // 检查是否已经在应用列表中
        const exists = this.appliedLoras.find(item => item.name === customName.trim());
        if (exists) {
            this.showError('This LoRA is already applied');
            return;
        }

        // Check if it's a Hugging Face LoRA (contains / and doesn't start with / or C:)
        const isHuggingFace = customName.includes('/') && !customName.startsWith('/') && !customName.match(/^[A-Za-z]:/);
        
        if (isHuggingFace) {
            // Try to cache the Hugging Face LoRA first
            try {
                // Show downloading progress indicator
                this.showDownloadProgress('Downloading Hugging Face LoRA...');
                
                // Parse repo_id and filename from the input
                let repo_id, filename;
                
                // Handle different input formats
                if (customName.includes('/blob/main/')) {
                    // Format: username/repo/blob/main/filename.safetensors
                    const parts = customName.trim().split('/');
                    repo_id = parts.slice(0, 2).join('/'); // username/repo
                    filename = parts[parts.length - 1]; // filename.safetensors
                } else if (customName.includes('/')) {
                    // Format: username/repo/filename.safetensors
                    const parts = customName.trim().split('/');
                    repo_id = parts.slice(0, -1).join('/');
                    filename = parts[parts.length - 1];
                } else {
                    throw new Error('Invalid Hugging Face LoRA format');
                }
                
                const response = await fetch(`${this.hostBase}/cache-hf-lora`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        repo_id: repo_id,
                        filename: filename,
                        display_name: customName.trim()
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to cache LoRA');
                }

                const result = await response.json();
                
                // Add to applied list with cached info
                this.appliedLoras.push({
                    name: customName.trim(),
                    weight: 1.0,
                    type: 'huggingface',
                    storedName: result.stored_name,
                    displayName: `${result.original_name} (Cached)`,
                    size: result.size,
                    timestamp: Math.floor(Date.now() / 1000)
                });

                this.renderAppliedLoras();
                this.updateApiCommand();
                this.hideDownloadProgress();
                this.showSuccess(`Hugging Face LoRA "${customName.trim()}" cached and added`);
                
                // Refresh the available LoRAs list to include the cached one
                this.loadAvailableLoras();
                
            } catch (error) {
                console.error('Error caching HF LoRA:', error);
                this.hideDownloadProgress();
                this.showError(`Failed to cache LoRA: ${error.message}`);
            }
        } else {
            // Local LoRA - add directly
            this.appliedLoras.push({
                name: customName.trim(),
                weight: 1.0,
                type: 'custom'
            });

            this.renderAppliedLoras();
            this.updateApiCommand();
            this.showSuccess(`Custom LoRA "${customName.trim()}" added`);
        }
    }

    async generateImage() {
        if (this.isGenerating) return;

        const prompt = this.getFormValues().prompt;

        this.isGenerating = true;
        this.showGenerationStatus(true);
        this.updateGenerateButton(true);

        try {
            let response;
            
            // Debug logging
            console.log('Generate image called');
            console.log('uploadedImageFile:', this.uploadedImageFile);
            console.log('serverUploadedImagePath:', this.serverUploadedImagePath);
            console.log('hasUploadedImage():', this.hasUploadedImage());

            if (this.hasUploadedImage()) {
                console.log('Using image-to-image generation');
                // Use image upload generation endpoint
                response = await this.generateImageWithUpload();
            } else {
                console.log('Using text-to-image generation');
                // Use regular text-to-image generation
                const params = this.getGenerationParams();
                response = await fetch(`${this.hostBase}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
            }

            const result = await this.handleFetchResponse(response);

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
        const endpoint = '/generate-with-image';
        const formData = this.buildFormData();

        if (this.serverUploadedImagePath) {
            formData.append('uploaded_image_path', this.serverUploadedImagePath);
        } else if (this.uploadedImageFile) {
            formData.append('image', this.uploadedImageFile);
        } else {
            throw new Error('No image selected or uploaded');
        }

        const response = await fetch(`${this.hostBase}${endpoint}`, {
            method: 'POST',
            body: formData
        });

        const result = await this.handleFetchResponse(response);

        console.log('Image-to-image generation completed');

        // Show single image
        const params = this.getGenerationParams();
        this.showSingleImage(result, params);
        this.showSuccess('Image generated successfully!');
    }

    async generateImageWithImage() {
        if (this.isGenerating) return;

        const prompt = this.getFormValues().prompt;
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
            const formData = this.buildFormData();

            // Always send direct image-to-image payload to the supported endpoint
            const endpoint = '/generate-with-image';
            formData.append('image', this.uploadedImageFile);

            const response = await fetch(`${this.hostBase}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            const result = await this.handleFetchResponse(response);
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
        const params = this.buildJsonPayload(true);

        // Always include LoRA configurations (empty list means remove any applied LoRA)
        params.loras = this.getLoraConfigs();

        // Ensure both models use the same seed
        if (!params.seed) {
            // Generate a random seed if none specified, so both models use the same one
            params.seed = Math.floor(Math.random() * 4294967295);
        }

        return params;
    }


    setupImageUpload() {
        const uploadArea = this.getElement('upload-area');
        const fileInput = this.getElement('image-upload');
        const removeBtn = this.getElement('remove-image');
        const uploadStatusLabel = this.getElement('upload-status-label');

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


        // Add event listener for upscale factor changes
        const upscaleFactorSelect = this.getElement('upscale-factor');
        if (upscaleFactorSelect) {
            upscaleFactorSelect.addEventListener('change', () => this.updateApiCommand());
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
        const uploadStatusLabel = this.getElement('upload-status-label');
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
        const uploadArea = this.getElement('upload-area');
        const uploadPlaceholder = this.getElement('upload-placeholder');
        const imagePreview = this.getElement('uploaded-image-preview');
        const uploadControls = this.getElement('upload-controls');

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
        const uploadArea = this.getElement('upload-area');
        const uploadPlaceholder = this.getElement('upload-placeholder');
        const imagePreview = this.getElement('uploaded-image-preview');
        const uploadControls = this.getElement('upload-controls');
        const fileInput = this.getElement('image-upload');

        // Clear file input
        if (fileInput) fileInput.value = '';

        // Hide preview and show placeholder
        if (uploadPlaceholder) uploadPlaceholder.classList.remove('hidden');
        if (imagePreview) imagePreview.classList.add('hidden');
        if (uploadControls) uploadControls.style.display = 'none';

        // Clear both image file and server path
        this.uploadedImageFile = null;
        this.serverUploadedImagePath = null;

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
        return (this.uploadedImageFile !== null && this.uploadedImageFile !== undefined) ||
               (this.serverUploadedImagePath !== null && this.serverUploadedImagePath !== undefined);
    }

    updateButtonVisibility() {
        const generateBtn = this.getElement('generate-btn');
        const generateWithImageBtn = this.getElement('generate-with-image-btn');

        if (this.hasUploadedImage()) {
            generateBtn.style.display = 'none';
            generateWithImageBtn.style.display = 'block';
        } else {
            generateBtn.style.display = 'block';
            generateWithImageBtn.style.display = 'none';
        }
    }

    showSingleImage(result, params) {
        const gallery = this.getElement('image-gallery');

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
        const modal = this.getElement('image-modal');
        const modalImage = this.getElement('modal-image');
        const modalPrompt = this.getElement('modal-prompt');
        const modalParams = this.getElement('modal-params');
        const modalTime = this.getElement('modal-time');

        modalImage.src = imageUrl;
        modalPrompt.textContent = params.prompt;
        modalParams.textContent = `${result.width || params.width}×${result.height || params.height}`;
        modalTime.textContent = result.generation_time || 'N/A';

        // Store current image data for download/copy
        this.currentModalData = { result, params, imageUrl };

        modal.classList.remove('hidden');
    }

    closeModal() {
        this.getElement('image-modal').classList.add('hidden');
        this.currentModalData = null;
    }

    showApiCommand() {
        const commandSection = this.getElement('api-command-section');
        const commandElement = this.getElement('api-command');

        // Build the JSON payload using utility function
        const jsonPayload = this.buildJsonPayload();

        // Escape the JSON string for shell
        const escapedJson = JSON.stringify(jsonPayload).replace(/"/g, '\\"');
        const command = `curl -s -X POST "${window.location.origin}/generate-and-return-image" -H "Content-Type: application/json" -d "${escapedJson}" -o "generated_image.png"`;

        commandElement.textContent = command;
        commandSection.classList.remove('hidden');
    }

    updateApiCommand() {
        const commandElement = this.getElement('api-command');
        const loadingElement = this.getElement('api-command-loading');
        if (!commandElement) return;

        // Show loading state briefly
        if (loadingElement) {
            loadingElement.style.display = 'flex';
            commandElement.style.display = 'none';
        }

        const values = this.getFormValues();
        const loras = this.getLoraConfigs();

        // Check if there's an uploaded image
        const uploadedImage = this.getElement('uploaded-image-preview');
        const hasImage = uploadedImage && !uploadedImage.classList.contains('hidden');

        let command;

        if (hasImage) {
            // Build command for generate-with-image-and-return endpoint
            // Show the filename but remind user to use full local path
            const imageFileName = this.uploadedImageFile ? this.uploadedImageFile.name : 'your_image.jpg';
            command = `curl -s -X POST "${window.location.origin}/generate-with-image-and-return" -F "image=@${imageFileName}" -F "prompt=${this.escapeForShell(values.prompt)}"`;

            if (values.negativePrompt) {
                command += ` -F "negative_prompt=${this.escapeForShell(values.negativePrompt)}"`;
            }

            if (values.seed) {
                command += ` -F "seed=${values.seed}"`;
            }

            command += ` -F "num_inference_steps=${values.numInferenceSteps}"`;
            command += ` -F "guidance_scale=${values.guidanceScale}"`;
            if (values.negativePrompt) {
                command += ` -F "true_cfg_scale=${values.trueCfgScale}"`;
            }

            if (loras && loras.length > 0) {
                // Convert LoRAs to JSON string for form data
                const lorasJson = JSON.stringify(loras.map(lora => ({
                    name: lora.name,
                    weight: parseFloat(lora.weight)
                })));
                command += ` -F "loras=${this.escapeForShell(lorasJson)}"`;
            }

            // Upscale parameters
            const upscaleCheckbox = this.getElement('upscale');
            if (upscaleCheckbox && upscaleCheckbox.checked) {
                command += ` -F "upscale=true"`;
                const upscaleFactorEl = this.getElement('upscale-factor');
                if (upscaleFactorEl && upscaleFactorEl.value) {
                    command += ` -F "upscale_factor=${upscaleFactorEl.value}"`;
                }
            }

            // Remove background flag
            const removeBgCheckbox = this.getElement('remove-background');
            if (removeBgCheckbox && removeBgCheckbox.checked) {
                command += ` -F "remove_background=true"`;
                const bgStrength = this.getElement('bg_strength');
                if (bgStrength && bgStrength.value) {
                    command += ` -F "bg_strength=${bgStrength.value}"`;
                }
            }

            // Downscale settings
            const downscaleCheckbox = this.getElement('downscale');
            if (downscaleCheckbox) {
                const downscaleValue = downscaleCheckbox.checked ? 'true' : 'false';
                command += ` -F "downscale=${downscaleValue}"`;
            }

            // Direct output to file - no need for jq or second curl
            command += ` -o "generated_image.png"`;
        } else {
            // Build command for generate-and-return-image-simple endpoint
            // Build the JSON payload first, then escape it properly for shell
            const jsonPayload = this.buildJsonPayload(true);

            // Escape the JSON string for shell
            const escapedJson = JSON.stringify(jsonPayload).replace(/"/g, '\\"');
            command = `curl -s -X POST "${window.location.origin}/generate-and-return-image" -H "Content-Type: application/json" -d "${escapedJson}" -o "generated_image.png"`;
        }

        commandElement.textContent = command;

        // Ensure the API command section is visible
        const commandSection = this.getElement('api-command-section');
        if (commandSection) {
            commandSection.classList.remove('hidden');
        }

        // Show/hide the help message for image-to-image commands
        const helpElement = this.getElement('api-command-help');
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

        const commandElement = this.getElement('api-command');
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
                    this.showSuccess('API Command Copied to Clipboard');
                })
                .catch((err) => {
                    console.error('Modern clipboard API failed:', err);
                    this.fallbackCopyToClipboard(textToCopy);
                });
        } else {
            console.error('Clipboard API not available');
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
                this.showSuccess('API Command Copied to Clipboard');
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



    showGenerationStatus(show) {
        const status = this.getElement('generation-status');
        const message = this.getElement('status-message');

        if (show) {
            message.textContent = 'Generating image...';
            status.classList.remove('hidden');
        } else {
            status.classList.add('hidden');
        }
    }

    showLoading(message) {
        const status = this.getElement('generation-status');
        const statusMessage = this.getElement('status-message');

        if (statusMessage) {
            statusMessage.textContent = message || 'Loading...';
        }
        if (status) {
            status.classList.remove('hidden');
        }
    }

    hideLoading() {
        const status = this.getElement('generation-status');
        if (status) {
            status.classList.add('hidden');
        }
    }

    updateGenerateButton(generating) {
        const generateBtn = this.getElement('generate-btn');
        const generateWithImageBtn = this.getElement('generate-with-image-btn');

        if (generating) {
            // Disable and dim the generate button
            if (generateBtn) {
                generateBtn.disabled = true;
                generateBtn.style.opacity = '0.5';
                generateBtn.style.cursor = 'not-allowed';
                generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            }
            
            // Disable and dim the generate with image button
            if (generateWithImageBtn) {
                generateWithImageBtn.disabled = true;
                generateWithImageBtn.style.opacity = '0.5';
                generateWithImageBtn.style.cursor = 'not-allowed';
                generateWithImageBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            }
        } else {
            // Re-enable and restore the generate button
            if (generateBtn) {
                generateBtn.disabled = false;
                generateBtn.style.opacity = '1';
                generateBtn.style.cursor = 'pointer';
                generateBtn.innerHTML = '<i class="fas fa-play"></i> Generate';
            }
            
            // Re-enable and restore the generate with image button
            if (generateWithImageBtn) {
                generateWithImageBtn.disabled = false;
                generateWithImageBtn.style.opacity = '1';
                generateWithImageBtn.style.cursor = 'pointer';
                generateWithImageBtn.innerHTML = '<i class="fas fa-image"></i> Generate';
            }
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

    showDownloadProgress(message) {
        // Create download progress indicator
        const progressContainer = document.createElement('div');
        progressContainer.id = 'download-progress';
        progressContainer.className = 'download-progress';
        progressContainer.innerHTML = `
            <div class="download-progress-content">
                <div class="download-progress-text">
                    <div class="download-filename">${message}</div>
                    <div class="download-status">Please wait...</div>
                </div>
                <div class="download-progress-bar">
                    <div class="download-progress-fill"></div>
                </div>
            </div>
        `;

        // Add to page
        document.body.appendChild(progressContainer);

        // Add slide-in animation
        setTimeout(() => {
            progressContainer.classList.add('download-progress-show');
        }, 100);

        // Store container reference
        this.downloadProgressContainer = progressContainer;
        this.downloadProgressFill = progressContainer.querySelector('.download-progress-fill');
        
        // Start animated progress bar
        this.animateDownloadProgress();
    }

    animateDownloadProgress() {
        if (!this.downloadProgressFill) return;
        
        let progress = 0;
        const animate = () => {
            if (this.downloadProgressContainer && this.downloadProgressFill) {
                progress += Math.random() * 10;
                if (progress > 90) progress = 90; // Don't go to 100% until actually done
                this.downloadProgressFill.style.width = progress + '%';
                this.downloadProgressTimeout = setTimeout(animate, 200);
            }
        };
        animate();
    }

    hideDownloadProgress() {
        const progressContainer = this.getElement('download-progress');
        if (progressContainer) {
            // Clear animation timeout
            if (this.downloadProgressTimeout) {
                clearTimeout(this.downloadProgressTimeout);
                this.downloadProgressTimeout = null;
            }
            
            // Show completion briefly
            progressContainer.classList.add('download-progress-complete');
            
            // Remove after animation
            setTimeout(() => {
                if (progressContainer.parentElement) {
                    progressContainer.remove();
                }
            }, 500);
        }

        // Clear references
        this.downloadProgressContainer = null;
        this.downloadProgressFill = null;
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
        const progressContainer = this.getElement('upload-progress');
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
        this.getElement('lora-file-input').click();
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

                // 上传成功后添加到可用列表（避免重复）
                const existingIndex = this.availableLoras.findIndex(lora => lora.storedName === uploadResult.filename);
                if (existingIndex === -1) {
                    this.availableLoras.push({
                        name: uploadResult.filename,
                        weight: 1.0,
                        type: 'uploaded',
                        storedName: uploadResult.filename,
                        displayName: `${uploadResult.filename} (Uploaded)`
                    });
                }

                this.populateLoraDropdown();
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
