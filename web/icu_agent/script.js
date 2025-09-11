// ICU Agent Frontend JavaScript
class ICUAgentApp {
    constructor() {
        this.currentPatient = null;
        this.isLeftSidebarCollapsed = false;
        this.isPatientSelectionOpen = false;
        this.apiBaseUrl = 'http://localhost:5000/api';
        this.samplePatients = [];
        this.workflowSteps = [];
        
        this.init();
    }

    init() {
        console.log('=== ICU Agent Frontend Initializing ===');
        console.log('Starting component initialization...');
        this.setupEventListeners();
        this.loadSampleData();
        this.setupMainContentDisabled();
        console.log('=== ICU Agent Frontend Initialization Complete ===');
    }

    setupEventListeners() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.bindEvents();
            });
        } else {
            this.bindEvents();
        }
    }

    bindEvents() {
        console.log('Binding events...');
        
        // Patient selection events
        const selectPatientBtn = document.getElementById('selectPatientBtn');
        const switchPatientBtn = document.getElementById('switchPatientBtn');
        const closePanelBtn = document.getElementById('closePanelBtn');
        const patientSearchInput = document.getElementById('patientSearchInput');

        if (selectPatientBtn) {
            var self = this;
            selectPatientBtn.addEventListener('click', function() {
                console.log('Select patient button clicked');
                try {
                    self.togglePatientSelection();
                } catch (error) {
                    console.error('Error in togglePatientSelection:', error);
                }
            });
        } else {
            console.error('selectPatientBtn not found');
        }

        if (switchPatientBtn) {
            var self = this;
            switchPatientBtn.addEventListener('click', function() {
                console.log('Switch patient button clicked');
                try {
                    self.togglePatientSelection();
                } catch (error) {
                    console.error('Error in switch patient togglePatientSelection:', error);
                }
            });
        }

        if (closePanelBtn) {
            var self = this;
            closePanelBtn.addEventListener('click', function() {
                console.log('Close panel button clicked');
                try {
                    self.closePatientSelection();
                } catch (error) {
                    console.error('Error in closePatientSelection:', error);
                }
            });
        }

        if (patientSearchInput) {
            var self = this;
            patientSearchInput.addEventListener('input', function(e) {
                self.filterPatients(e.target.value);
            });
        }

        // Sidebar toggle
        const toggleLeftSidebar = document.getElementById('toggleLeftSidebar');
        if (toggleLeftSidebar) {
            var self = this;
            toggleLeftSidebar.addEventListener('click', function() {
                self.toggleLeftSidebar();
            });
        }

        // Chat functionality
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');

        if (chatInput) {
            var self = this;
            chatInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    self.sendMessage();
                }
            });

            chatInput.addEventListener('input', function() {
                self.updateSendButtonState();
            });
        }

        if (sendBtn) {
            var self = this;
            sendBtn.addEventListener('click', function() {
                self.sendMessage();
            });
        }

        console.log('Events bound successfully');
    }

    loadSampleData() {
        console.log('--- Loading Sample Data ---');
        // Sample patients data
        this.samplePatients = [
            {
                id: 1,
                name: '张某某',
                bedNumber: 'A-101',
                riskLevel: 'medium',
                age: 65,
                condition: '急性心肌梗死',
                avatar: '张'
            },
            {
                id: 2,
                name: '李某某',
                bedNumber: 'A-102',
                riskLevel: 'high',
                age: 72,
                condition: '急性呼吸窘迫综合征',
                avatar: '李'
            },
            {
                id: 3,
                name: '王某某',
                bedNumber: 'B-201',
                riskLevel: 'low',
                age: 58,
                condition: '糖尿病酮症酸中毒',
                avatar: '王'
            },
            {
                id: 4,
                name: '陈某某',
                bedNumber: 'B-202',
                riskLevel: 'medium',
                age: 69,
                condition: '脑出血',
                avatar: '陈'
            },
            {
                id: 5,
                name: '刘某某',
                bedNumber: 'C-301',
                riskLevel: 'high',
                age: 75,
                condition: '感染性休克',
                avatar: '刘'
            }
        ];

        this.renderPatientList();
        console.log('Sample data loaded successfully. Total patients:', this.samplePatients.length);
        console.log('Patient list:', this.samplePatients.map(function(p) { return p.name + ' (' + p.bedNumber + ')'; }).join(', '));
    }

    setupMainContentDisabled() {
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.classList.add('disabled');
        }
    }

    togglePatientSelection() {
        console.log('togglePatientSelection called');
        const panel = document.getElementById('patientSelectionPanel');
        if (!panel) {
            console.error('patientSelectionPanel not found');
            return;
        }
        
        const isOpen = panel.classList.contains('active');
        console.log('Panel is currently:', isOpen ? 'open' : 'closed');
        
        if (isOpen) {
            this.closePatientSelection();
        } else {
            this.openPatientSelection();
        }
    }

    openPatientSelection() {
        console.log('Opening patient selection panel');
        const panel = document.getElementById('patientSelectionPanel');
        if (!panel) {
            console.error('patientSelectionPanel not found in openPatientSelection');
            return;
        }
        panel.classList.remove('hidden');
        panel.classList.add('active');
        this.isPatientSelectionOpen = true;
        
        // Focus on search input
        const searchInput = document.getElementById('patientSearchInput');
        if (searchInput) {
            setTimeout(function() {
                searchInput.focus();
            }, 300);
        }
    }

    closePatientSelection() {
        console.log('Closing patient selection panel');
        const panel = document.getElementById('patientSelectionPanel');
        if (!panel) {
            console.error('patientSelectionPanel not found in closePatientSelection');
            return;
        }
        panel.classList.remove('active');
        panel.classList.add('hidden');
        this.isPatientSelectionOpen = false;
    }

    renderPatientList(filteredPatients) {
        const patientList = document.getElementById('patientList');
        if (!patientList) {
            console.error('patientList element not found');
            return;
        }
        
        const patients = filteredPatients || this.samplePatients;
        const self = this;
        
        const html = patients.map(function(patient) {
            return '<div class="patient-card" onclick="icuApp.selectPatient(' + patient.id + ')">' +
                '<div class="patient-avatar"><span>' + patient.avatar + '</span></div>' +
                '<div class="patient-card-info">' +
                '<h4>' + patient.name + '</h4>' +
                '<div class="patient-card-meta">' +
                '<span>床号: ' + patient.bedNumber + '</span>' +
                '<span class="risk-level ' + patient.riskLevel + '">' + self.getRiskLevelText(patient.riskLevel) + '</span>' +
                '<span>' + patient.age + '岁</span>' +
                '<span>' + patient.condition + '</span>' +
                '</div></div></div>';
        }).join('');
        
        patientList.innerHTML = html;
        console.log('Patient list rendered with', patients.length, 'patients');
    }

    filterPatients(query) {
        if (!query.trim()) {
            this.renderPatientList();
            return;
        }

        const filtered = this.samplePatients.filter(patient => 
            patient.name.includes(query) || 
            patient.bedNumber.includes(query) ||
            patient.condition.includes(query)
        );
        
        this.renderPatientList(filtered);
    }

    selectPatient(patientId) {
        console.log('--- Selecting Patient ---');
        console.log('Requested patient ID:', patientId);
        
        const patient = this.samplePatients.find(function(p) { return p.id === patientId; });
        if (!patient) {
            console.error('Patient not found with ID:', patientId);
            return;
        }

        console.log('Found patient:', patient.name, '(' + patient.bedNumber + ')');
        this.currentPatient = patient;
        
        console.log('Updating patient hub interface...');
        this.updatePatientHub();
        
        console.log('Closing patient selection panel...');
        this.closePatientSelection();
        
        console.log('Enabling main content area...');
        this.enableMainContent();
        
        console.log('Loading patient-specific data...');
        this.loadPatientData();
        
        console.log('=== Patient Selection Complete ===');
        console.log('Current patient:', this.currentPatient.name, '-', this.currentPatient.condition);
    }

    updatePatientHub() {
        console.log('Updating patient hub display...');
        
        const emptyState = document.getElementById('patientHubEmpty');
        const selectedState = document.getElementById('patientHubSelected');
        
        if (emptyState) {
            emptyState.classList.add('hidden');
            console.log('Hidden empty state');
        }
        if (selectedState) {
            selectedState.classList.remove('hidden');
            console.log('Showed selected state');
        }
        
        // Update patient info
        console.log('Updating patient display elements...');
        document.getElementById('patientInitials').textContent = this.currentPatient.avatar;
        document.getElementById('patientName').textContent = this.currentPatient.name;
        document.getElementById('patientBed').textContent = '床号: ' + this.currentPatient.bedNumber;
        document.getElementById('patientRisk').textContent = this.getRiskLevelText(this.currentPatient.riskLevel);
        document.getElementById('patientRisk').className = 'risk-level ' + this.currentPatient.riskLevel;
        console.log('Patient hub updated with:', this.currentPatient.name, '-', this.getRiskLevelText(this.currentPatient.riskLevel));
    }

    enableMainContent() {
        console.log('Enabling main content interface...');
        
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.classList.remove('disabled');
            console.log('Main content area enabled');
        }
        
        // Enable chat input
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.disabled = false;
            chatInput.placeholder = '请输入您的问题...';
            console.log('Chat input enabled');
        }
        
        this.updateSendButtonState();
        this.updateChatStatus('等待输入...');
        console.log('Main content activation complete');
    }

    toggleLeftSidebar() {
        const sidebar = document.getElementById('leftSidebar');
        this.isLeftSidebarCollapsed = !this.isLeftSidebarCollapsed;
        
        if (this.isLeftSidebarCollapsed) {
            sidebar.classList.add('collapsed');
        } else {
            sidebar.classList.remove('collapsed');
        }
    }

    loadPatientData() {
        console.log('--- Loading Patient-Specific Data ---');
        if (!this.currentPatient) {
            console.error('No current patient selected');
            return;
        }
        
        console.log('Loading data for:', this.currentPatient.name);
        
        // Load risk data
        console.log('Loading risk assessment data...');
        this.loadRiskData();
        
        // Load events data
        console.log('Loading events timeline...');
        this.loadEventsData();
        
        // Load chat history
        console.log('Initializing chat interface...');
        this.loadChatHistory();
        
        console.log('=== Patient Data Loading Complete ===');
    }

    loadRiskData() {
        const riskData = [
            { name: '急性肾损伤', probability: 85, level: 'high' },
            { name: '感染性休克', probability: 65, level: 'medium' },
            { name: '心律失常', probability: 45, level: 'medium' },
            { name: '呼吸衰竭', probability: 25, level: 'low' }
        ];

        console.log('Risk data generated:', riskData.length, 'risk factors');
        riskData.forEach(function(risk) {
            console.log('  - ' + risk.name + ': ' + risk.probability + '% (' + risk.level + ')');
        });

        this.renderRiskList(riskData);
        this.updateRiskIndicator(riskData.length);
        console.log('Risk assessment interface updated');
    }

    renderRiskList(risks) {
        const riskList = document.getElementById('riskList');
        if (!riskList) {
            console.error('riskList element not found');
            return;
        }
        
        const html = risks.map(function(risk) {
            return '<div class="risk-item">' +
                '<span class="risk-item-name">' + risk.name + '</span>' +
                '<span class="risk-probability ' + risk.level + '">' + risk.probability + '%</span>' +
                '</div>';
        }).join('');
        
        riskList.innerHTML = html;
        console.log('Risk list rendered with', risks.length, 'items');
    }

    updateRiskIndicator(count) {
        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = count;
        
        const riskIndicator = document.getElementById('riskIndicator');
        var level = count > 2 ? 'high' : count > 0 ? 'medium' : 'low';
        riskIndicator.className = 'icon-indicator risk-indicator ' + level;
    }

    loadEventsData() {
        const eventsData = [
            {
                time: '14:32',
                title: '血压异常',
                description: '收缩压升高至180mmHg',
                type: 'warning'
            },
            {
                time: '13:45',
                title: '用药记录',
                description: '静脉注射呋塞米20mg',
                type: 'info'
            },
            {
                time: '12:20',
                title: '检验结果',
                description: '肌酐值上升至2.1mg/dL',
                type: 'critical'
            },
            {
                time: '11:30',
                title: '体征监测',
                description: '心率102次/分，呼吸20次/分',
                type: 'normal'
            }
        ];

        console.log('Events data generated:', eventsData.length, 'events');
        eventsData.forEach(function(event) {
            console.log('  - ' + event.time + ' ' + event.title + ' (' + event.type + ')');
        });

        this.renderEventsList(eventsData);
        var hasAlerts = eventsData.some(function(e) { return e.type === 'critical' || e.type === 'warning'; });
        this.updateEventIndicator(hasAlerts);
        console.log('Events timeline interface updated, alerts:', hasAlerts);
    }

    renderEventsList(events) {
        const eventsTimeline = document.getElementById('eventsTimeline');
        if (!eventsTimeline) {
            console.error('eventsTimeline element not found');
            return;
        }
        
        const html = events.map(function(event) {
            return '<div class="event-item" title="' + event.description + '">' +
                '<div class="event-time">' + event.time + '</div>' +
                '<div class="event-content">' +
                '<div class="event-title">' + event.title + '</div>' +
                '<div class="event-description">' + event.description + '</div>' +
                '</div></div>';
        }).join('');
        
        eventsTimeline.innerHTML = html;
        console.log('Events list rendered with', events.length, 'items');
    }

    updateEventIndicator(hasNewEvents) {
        const eventDot = document.getElementById('eventDot');
        if (hasNewEvents) {
            eventDot.style.display = 'block';
        } else {
            eventDot.style.display = 'none';
        }
    }

    loadChatHistory() {
        // Clear existing messages except welcome
        const chatMessages = document.getElementById('chatMessages');
        const welcomeMessage = chatMessages.querySelector('.welcome-message');
        chatMessages.innerHTML = '';
        
        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }
        
        // Add initial patient context message
        this.addAgentMessage('已为您加载患者' + this.currentPatient.name + '的医疗档案。该患者目前在' + 
            this.currentPatient.bedNumber + '床，诊断为' + this.currentPatient.condition + 
            '。您可以询问有关患者的病情、治疗方案或风险评估等问题。');
    }

    updateChatStatus(status) {
        const chatStatus = document.getElementById('chatStatus');
        const statusText = chatStatus.querySelector('.status-text');
        statusText.textContent = status;
    }

    updateSendButtonState() {
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const hasText = chatInput.value.trim().length > 0;
        
        sendBtn.disabled = !hasText || chatInput.disabled;
    }

    async sendMessage() {
        console.log('--- Sending Message ---');
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();
        
        console.log('User message:', message);
        
        if (!message) {
            console.warn('Empty message, ignoring');
            return;
        }
        
        if (!this.currentPatient) {
            console.error('No current patient selected');
            return;
        }
        
        console.log('Processing message for patient:', this.currentPatient.name);
        
        // Add user message
        this.addUserMessage(message);
        chatInput.value = '';
        this.updateSendButtonState();
        
        // Show thinking state
        console.log('Activating Agent thinking state...');
        this.showAgentThinking();
        this.showWorkflowActive();
        
        try {
            console.log('Sending request to backend API...');
            // Send to backend
            const response = await this.callBackendAPI('/chat', {
                patient_id: this.currentPatient.id,
                message: message
            });
            
            console.log('Received response from backend');
            // Hide thinking state
            this.hideAgentThinking();
            
            // Add agent response
            this.addAgentMessage(response.response);
            
            // Show workflow completion
            this.showWorkflowCompleted();
            console.log('=== Message Processing Complete ===');
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideAgentThinking();
            this.addAgentMessage('抱歉，处理您的问题时遇到了错误。请稍后重试。');
            this.showWorkflowError();
        }
    }

    addUserMessage(message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageElement = document.createElement('div');
        messageElement.className = 'user-message';
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
        
        chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    }

    addAgentMessage(message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageElement = document.createElement('div');
        messageElement.className = 'agent-message';
        messageElement.innerHTML = `
            <div class="agent-avatar">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 12L11 14L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="2"/>
                </svg>
            </div>
            <div class="message-content">
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
        
        chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    }

    showAgentThinking() {
        const chatMessages = document.getElementById('chatMessages');
        const thinkingElement = document.createElement('div');
        thinkingElement.className = 'agent-message thinking-message';
        thinkingElement.innerHTML = `
            <div class="agent-avatar">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 12L11 14L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="2"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="thinking-indicator">
                    <span>正在分析</span>
                    <div class="thinking-dots">
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                        <div class="thinking-dot"></div>
                    </div>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(thinkingElement);
        this.scrollToBottom();
        this.updateChatStatus('Agent 正在思考...');
    }

    hideAgentThinking() {
        const thinkingMessage = document.querySelector('.thinking-message');
        if (thinkingMessage) {
            thinkingMessage.remove();
        }
        this.updateChatStatus('等待输入...');
    }

    showWorkflowActive() {
        const workflowStatus = document.getElementById('workflowStatus');
        workflowStatus.textContent = '正在处理';
        workflowStatus.className = 'workflow-status active';
        
        // Simulate workflow steps
        this.simulateWorkflowSteps();
    }

    showWorkflowCompleted() {
        const workflowStatus = document.getElementById('workflowStatus');
        workflowStatus.textContent = '已完成';
        workflowStatus.className = 'workflow-status';
        
        setTimeout(() => {
            workflowStatus.textContent = '待命中';
            this.clearWorkflowSteps();
        }, 3000);
    }

    showWorkflowError() {
        const workflowStatus = document.getElementById('workflowStatus');
        workflowStatus.textContent = '处理错误';
        workflowStatus.className = 'workflow-status';
        
        setTimeout(() => {
            workflowStatus.textContent = '待命中';
            this.clearWorkflowSteps();
        }, 3000);
    }

    simulateWorkflowSteps() {
        const steps = [
            '解析问题意图',
            '检索患者档案',
            '分析医疗数据',
            '调用诊断模型',
            '生成回复内容'
        ];
        
        const workflowSteps = document.getElementById('workflowSteps');
        workflowSteps.innerHTML = '';
        
        steps.forEach((step, index) => {
            setTimeout(() => {
                this.addWorkflowStep(step, index, steps.length);
            }, index * 800);
        });
    }

    addWorkflowStep(stepText, index, total) {
        const workflowSteps = document.getElementById('workflowSteps');
        const stepElement = document.createElement('div');
        stepElement.className = 'workflow-step active';
        stepElement.innerHTML = `
            <div class="step-spinner"></div>
            <div class="step-text">${stepText}...</div>
        `;
        
        workflowSteps.appendChild(stepElement);
        
        // Mark as completed after delay
        setTimeout(() => {
            stepElement.className = 'workflow-step completed';
            stepElement.innerHTML = `
                <svg class="step-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 12L11 14L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                </svg>
                <div class="step-text">${stepText}</div>
            `;
        }, 600);
    }

    clearWorkflowSteps() {
        const workflowSteps = document.getElementById('workflowSteps');
        workflowSteps.innerHTML = `
            <div class="workflow-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>
                    <path d="M19.4 15A1.65 1.65 0 0 0 21 13.35V10.65A1.65 1.65 0 0 0 19.4 9L17.6 7.5A7.81 7.81 0 0 0 16 6.5L15.28 4.6A1.65 1.65 0 0 0 13.93 3.5H10.07A1.65 1.65 0 0 0 8.72 4.6L8 6.5A7.81 7.81 0 0 0 6.4 7.5L4.6 9A1.65 1.65 0 0 0 3 10.65V13.35A1.65 1.65 0 0 0 4.6 15L6.4 16.5A7.81 7.81 0 0 0 8 17.5L8.72 19.4A1.65 1.65 0 0 0 10.07 20.5H13.93A1.65 1.65 0 0 0 15.28 19.4L16 17.5A7.81 7.81 0 0 0 17.6 16.5Z" stroke="currentColor" stroke-width="2"/>
                </svg>
                <p>Agent 正在待命中</p>
                <span>当您发送问题时，这里将显示 Agent 的思考过程</span>
            </div>
        `;
    }

    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async callBackendAPI(endpoint, data) {
        const response = await fetch(this.apiBaseUrl + endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
        }
        
        return await response.json();
    }

    getRiskLevelText(level) {
        const levels = {
            'low': '低风险',
            'medium': '中风险',
            'high': '高风险'
        };
        return levels[level] || level;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application
const icuApp = new ICUAgentApp();

// Make it globally accessible for onclick handlers
window.icuApp = icuApp;
