// ICU Agent Frontend JavaScript - 兼容版本
// 避免使用ES6 class语法，提高浏览器兼容性

var ICUAgent = {
    currentPatient: null,
    isLeftSidebarCollapsed: false,
    isPatientSelectionOpen: false,
    apiBaseUrl: 'http://localhost:5000/api',
    samplePatients: [],
    
    // 初始化
    init: function() {
        console.log('ICU Agent Frontend initialized');
        this.setupEventListeners();
        this.loadSampleData();
        this.setupMainContentDisabled();
    },
    
    // 设置事件监听器
    setupEventListeners: function() {
        var self = this;
        
        // 等待DOM加载完成
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                self.bindEvents();
            });
        } else {
            this.bindEvents();
        }
    },
    
    // 绑定事件
    bindEvents: function() {
        var self = this;
        console.log('Binding events...');
        
        // 病人选择事件
        var selectPatientBtn = document.getElementById('selectPatientBtn');
        var switchPatientBtn = document.getElementById('switchPatientBtn');
        var closePanelBtn = document.getElementById('closePanelBtn');
        var patientSearchInput = document.getElementById('patientSearchInput');
        
        if (selectPatientBtn) {
            selectPatientBtn.addEventListener('click', function() {
                console.log('Select patient button clicked');
                self.togglePatientSelection();
            });
        } else {
            console.error('selectPatientBtn not found');
        }
        
        if (switchPatientBtn) {
            switchPatientBtn.addEventListener('click', function() {
                console.log('Switch patient button clicked');
                self.togglePatientSelection();
            });
        }
        
        if (closePanelBtn) {
            closePanelBtn.addEventListener('click', function() {
                console.log('Close panel button clicked');
                self.closePatientSelection();
            });
        }
        
        if (patientSearchInput) {
            patientSearchInput.addEventListener('input', function(e) {
                self.filterPatients(e.target.value);
            });
        }
        
        // 侧边栏切换
        var toggleLeftSidebar = document.getElementById('toggleLeftSidebar');
        if (toggleLeftSidebar) {
            toggleLeftSidebar.addEventListener('click', function() {
                self.toggleLeftSidebar();
            });
        }
        
        // 聊天功能
        var chatInput = document.getElementById('chatInput');
        var sendBtn = document.getElementById('sendBtn');
        
        if (chatInput) {
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
            sendBtn.addEventListener('click', function() {
                self.sendMessage();
            });
        }
        
        console.log('Events bound successfully');
    },
    
    // 加载示例数据
    loadSampleData: function() {
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
        console.log('Sample data loaded, patients:', this.samplePatients.length);
    },
    
    // 设置主内容区禁用状态
    setupMainContentDisabled: function() {
        var mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.classList.add('disabled');
        }
    },
    
    // 切换病人选择面板
    togglePatientSelection: function() {
        console.log('togglePatientSelection called');
        var panel = document.getElementById('patientSelectionPanel');
        if (!panel) {
            console.error('patientSelectionPanel not found');
            return;
        }
        
        var isOpen = panel.classList.contains('active');
        console.log('Panel is currently:', isOpen ? 'open' : 'closed');
        
        if (isOpen) {
            this.closePatientSelection();
        } else {
            this.openPatientSelection();
        }
    },
    
    // 打开病人选择面板
    openPatientSelection: function() {
        console.log('Opening patient selection panel');
        var panel = document.getElementById('patientSelectionPanel');
        if (panel) {
            panel.classList.remove('hidden');
            panel.classList.add('active');
            this.isPatientSelectionOpen = true;
            
            // 聚焦搜索框
            var searchInput = document.getElementById('patientSearchInput');
            if (searchInput) {
                setTimeout(function() {
                    searchInput.focus();
                }, 300);
            }
        }
    },
    
    // 关闭病人选择面板
    closePatientSelection: function() {
        console.log('Closing patient selection panel');
        var panel = document.getElementById('patientSelectionPanel');
        if (panel) {
            panel.classList.remove('active');
            panel.classList.add('hidden');
            this.isPatientSelectionOpen = false;
        }
    },
    
    // 渲染病人列表
    renderPatientList: function(filteredPatients) {
        var patientList = document.getElementById('patientList');
        if (!patientList) {
            console.error('patientList element not found');
            return;
        }
        
        var patients = filteredPatients || this.samplePatients;
        var self = this;
        
        var html = patients.map(function(patient) {
            return '<div class="patient-card" onclick="ICUAgent.selectPatient(' + patient.id + ')">' +
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
    },
    
    // 筛选病人
    filterPatients: function(query) {
        if (!query || !query.trim()) {
            this.renderPatientList();
            return;
        }
        
        var filtered = this.samplePatients.filter(function(patient) {
            return patient.name.indexOf(query) !== -1 || 
                   patient.bedNumber.indexOf(query) !== -1 ||
                   patient.condition.indexOf(query) !== -1;
        });
        
        this.renderPatientList(filtered);
    },
    
    // 选择病人
    selectPatient: function(patientId) {
        console.log('Selecting patient:', patientId);
        
        var patient = this.samplePatients.find(function(p) {
            return p.id === patientId;
        });
        
        if (!patient) {
            console.error('Patient not found:', patientId);
            return;
        }
        
        this.currentPatient = patient;
        this.updatePatientHub();
        this.closePatientSelection();
        this.enableMainContent();
        this.loadPatientData();
        
        console.log('Patient selected:', patient);
    },
    
    // 更新病人中心栏
    updatePatientHub: function() {
        var emptyState = document.getElementById('patientHubEmpty');
        var selectedState = document.getElementById('patientHubSelected');
        
        if (emptyState) emptyState.classList.add('hidden');
        if (selectedState) selectedState.classList.remove('hidden');
        
        // 更新病人信息
        var elements = {
            'patientInitials': this.currentPatient.avatar,
            'patientName': this.currentPatient.name,
            'patientBed': '床号: ' + this.currentPatient.bedNumber,
            'patientRisk': this.getRiskLevelText(this.currentPatient.riskLevel)
        };
        
        for (var id in elements) {
            var element = document.getElementById(id);
            if (element) {
                element.textContent = elements[id];
            }
        }
        
        var riskElement = document.getElementById('patientRisk');
        if (riskElement) {
            riskElement.className = 'risk-level ' + this.currentPatient.riskLevel;
        }
    },
    
    // 启用主内容区
    enableMainContent: function() {
        var mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.classList.remove('disabled');
        }
        
        // 启用聊天输入
        var chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.disabled = false;
            chatInput.placeholder = '请输入您的问题...';
        }
        
        this.updateSendButtonState();
        this.updateChatStatus('等待输入...');
    },
    
    // 切换左侧边栏
    toggleLeftSidebar: function() {
        var sidebar = document.getElementById('leftSidebar');
        if (!sidebar) return;
        
        this.isLeftSidebarCollapsed = !this.isLeftSidebarCollapsed;
        
        if (this.isLeftSidebarCollapsed) {
            sidebar.classList.add('collapsed');
        } else {
            sidebar.classList.remove('collapsed');
        }
    },
    
    // 加载病人数据
    loadPatientData: function() {
        if (!this.currentPatient) return;
        
        this.loadRiskData();
        this.loadEventsData();
        this.loadChatHistory();
    },
    
    // 加载风险数据
    loadRiskData: function() {
        var riskData = [
            { name: '急性肾损伤', probability: 85, level: 'high' },
            { name: '感染性休克', probability: 65, level: 'medium' },
            { name: '心律失常', probability: 45, level: 'medium' },
            { name: '呼吸衰竭', probability: 25, level: 'low' }
        ];
        
        this.renderRiskList(riskData);
        this.updateRiskIndicator(riskData.length);
    },
    
    // 渲染风险列表
    renderRiskList: function(risks) {
        var riskList = document.getElementById('riskList');
        if (!riskList) return;
        
        var html = risks.map(function(risk) {
            return '<div class="risk-item">' +
                '<span class="risk-item-name">' + risk.name + '</span>' +
                '<span class="risk-probability ' + risk.level + '">' + risk.probability + '%</span>' +
                '</div>';
        }).join('');
        
        riskList.innerHTML = html;
    },
    
    // 更新风险指示器
    updateRiskIndicator: function(count) {
        var riskBadge = document.getElementById('riskBadge');
        if (riskBadge) {
            riskBadge.textContent = count;
        }
        
        var riskIndicator = document.getElementById('riskIndicator');
        if (riskIndicator) {
            var level = count > 2 ? 'high' : count > 0 ? 'medium' : 'low';
            riskIndicator.className = 'icon-indicator risk-indicator ' + level;
        }
    },
    
    // 加载事件数据
    loadEventsData: function() {
        var eventsData = [
            { time: '14:32', title: '血压异常', description: '收缩压升高至180mmHg', type: 'warning' },
            { time: '13:45', title: '用药记录', description: '静脉注射呋塞米20mg', type: 'info' },
            { time: '12:20', title: '检验结果', description: '肌酐值上升至2.1mg/dL', type: 'critical' },
            { time: '11:30', title: '体征监测', description: '心率102次/分，呼吸20次/分', type: 'normal' }
        ];
        
        this.renderEventsList(eventsData);
        this.updateEventIndicator(eventsData.some(function(e) {
            return e.type === 'critical' || e.type === 'warning';
        }));
    },
    
    // 渲染事件列表
    renderEventsList: function(events) {
        var eventsTimeline = document.getElementById('eventsTimeline');
        if (!eventsTimeline) return;
        
        var html = events.map(function(event) {
            return '<div class="event-item" title="' + event.description + '">' +
                '<div class="event-time">' + event.time + '</div>' +
                '<div class="event-content">' +
                '<div class="event-title">' + event.title + '</div>' +
                '<div class="event-description">' + event.description + '</div>' +
                '</div></div>';
        }).join('');
        
        eventsTimeline.innerHTML = html;
    },
    
    // 更新事件指示器
    updateEventIndicator: function(hasNewEvents) {
        var eventDot = document.getElementById('eventDot');
        if (eventDot) {
            eventDot.style.display = hasNewEvents ? 'block' : 'none';
        }
    },
    
    // 加载聊天历史
    loadChatHistory: function() {
        var chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        var welcomeMessage = chatMessages.querySelector('.welcome-message');
        chatMessages.innerHTML = '';
        
        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }
        
        // 添加初始病人上下文消息
        this.addAgentMessage('已为您加载患者' + this.currentPatient.name + '的医疗档案。该患者目前在' + 
            this.currentPatient.bedNumber + '床，诊断为' + this.currentPatient.condition + 
            '。您可以询问有关患者的病情、治疗方案或风险评估等问题。');
    },
    
    // 更新聊天状态
    updateChatStatus: function(status) {
        var chatStatus = document.getElementById('chatStatus');
        if (chatStatus) {
            var statusText = chatStatus.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = status;
            }
        }
    },
    
    // 更新发送按钮状态
    updateSendButtonState: function() {
        var chatInput = document.getElementById('chatInput');
        var sendBtn = document.getElementById('sendBtn');
        if (!chatInput || !sendBtn) return;
        
        var hasText = chatInput.value.trim().length > 0;
        sendBtn.disabled = !hasText || chatInput.disabled;
    },
    
    // 发送消息
    sendMessage: function() {
        console.log('Send message called');
        var chatInput = document.getElementById('chatInput');
        if (!chatInput) return;
        
        var message = chatInput.value.trim();
        if (!message || !this.currentPatient) return;
        
        // 添加用户消息
        this.addUserMessage(message);
        chatInput.value = '';
        this.updateSendButtonState();
        
        // 显示思考状态
        this.showAgentThinking();
        this.showWorkflowActive();
        
        // 模拟API调用
        var self = this;
        setTimeout(function() {
            self.hideAgentThinking();
            self.addAgentMessage('根据患者' + self.currentPatient.name + '的病情分析，建议您密切关注患者的生命体征变化。基于当前的临床数据，我建议进行进一步的评估。');
            self.showWorkflowCompleted();
        }, 2000);
    },
    
    // 添加用户消息
    addUserMessage: function(message) {
        var chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        var messageElement = document.createElement('div');
        messageElement.className = 'user-message';
        messageElement.innerHTML = '<div class="message-content"><p>' + this.escapeHtml(message) + '</p></div>';
        
        chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    },
    
    // 添加Agent消息
    addAgentMessage: function(message) {
        var chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        var messageElement = document.createElement('div');
        messageElement.className = 'agent-message';
        messageElement.innerHTML = 
            '<div class="agent-avatar">🤖</div>' +
            '<div class="message-content"><p>' + this.escapeHtml(message) + '</p></div>';
        
        chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    },
    
    // 显示Agent思考
    showAgentThinking: function() {
        var chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        var thinkingElement = document.createElement('div');
        thinkingElement.className = 'agent-message thinking-message';
        thinkingElement.innerHTML = 
            '<div class="agent-avatar">🤖</div>' +
            '<div class="message-content">' +
            '<div class="thinking-indicator">' +
            '<span>正在分析</span>' +
            '<div class="thinking-dots">' +
            '<div class="thinking-dot"></div>' +
            '<div class="thinking-dot"></div>' +
            '<div class="thinking-dot"></div>' +
            '</div></div></div>';
        
        chatMessages.appendChild(thinkingElement);
        this.scrollToBottom();
        this.updateChatStatus('Agent 正在思考...');
    },
    
    // 隐藏Agent思考
    hideAgentThinking: function() {
        var thinkingMessage = document.querySelector('.thinking-message');
        if (thinkingMessage) {
            thinkingMessage.remove();
        }
        this.updateChatStatus('等待输入...');
    },
    
    // 显示工作流激活
    showWorkflowActive: function() {
        var workflowStatus = document.getElementById('workflowStatus');
        if (workflowStatus) {
            workflowStatus.textContent = '正在处理';
            workflowStatus.className = 'workflow-status active';
        }
    },
    
    // 显示工作流完成
    showWorkflowCompleted: function() {
        var workflowStatus = document.getElementById('workflowStatus');
        if (workflowStatus) {
            workflowStatus.textContent = '已完成';
            workflowStatus.className = 'workflow-status';
            
            var self = this;
            setTimeout(function() {
                workflowStatus.textContent = '待命中';
            }, 3000);
        }
    },
    
    // 滚动到底部
    scrollToBottom: function() {
        var chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    },
    
    // 获取风险等级文本
    getRiskLevelText: function(level) {
        var levels = {
            'low': '低风险',
            'medium': '中风险',
            'high': '高风险'
        };
        return levels[level] || level;
    },
    
    // HTML转义
    escapeHtml: function(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// 初始化应用
console.log('ICU Agent script loaded');
ICUAgent.init();

// 全局暴露，供onclick使用
window.ICUAgent = ICUAgent;
