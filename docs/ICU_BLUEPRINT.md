# ICU多Agent系统蓝图

## 1. 数据分析和理解

### 1.1 ICU患者数据结构分析

通过对`database/icu_patients/`文件夹中的患者数据进行分析，发现以下关键特征：

#### 1.1.1 数据格式
- **文件命名**: 以患者ID命名（如`1125200959.json`）
- **数据结构**: JSON格式，包含`patient_id`、`meta_id`、`sequence`、`static`、`error_event`等字段

#### 1.1.2 事件序列（sequence）
每个患者的事件序列包含以下关键信息：
- **事件ID**: 唯一标识符
- **时间戳**: ISO格式时间
- **事件类型**: `surgery`、`order`、`lab`、`history`、`nursing`等
- **子类型**: 具体的事件分类
- **事件内容**: 详细的医疗信息
- **风险标记**: 当前为空数组，可用于风险标注
- **标志位**: 用于事件状态标记
- **元数据**: 额外信息

#### 1.1.3 事件类型分析
1. **手术事件** (`surgery`): 如"经导管颅内血管弹簧圈栓塞术"
2. **医嘱事件** (`order`): 包括临时医嘱和长期医嘱
3. **实验室检查** (`lab`): 各种检验结果
4. **病史记录** (`history`): 病程记录、诊断等
5. **护理事件** (`nursing`): 护理措施、生命体征监测等

#### 1.1.4 静态信息
- **诊断记录**: ICD-10编码和诊断名称
- **病史**: 患者既往病史

### 1.2 风险分类体系

基于`risks_table.json`，ICU风险分为以下主要系统：

1. **循环系统**: 休克、心律失常、急性缺血、高血压并发症等
2. **呼吸系统**: 氧合/通气衰竭、下呼吸道感染、气道事件等
3. **消化系统**: 消化道出血、肝胆胰疾病、胃肠疾病等
4. **泌尿系统**: 感染、肾功能损伤、泌尿系统疾病等
5. **神经系统**: 脑血管事件、颅压与脑疝、癫痫等
6. **内分泌与免疫**: 血糖异常、甲状腺疾病、电解质紊乱等
7. **妇产科**: 产科急症、妇科疾病等
8. **血液/凝血系统**: 出凝血异常、血小板疾病、贫血等
9. **创伤与感染**: 创伤、骨与四肢疾病、感染等

## 2. 系统需求分析

### 2.1 核心功能需求

#### 2.1.1 风险预测功能
- **实时风险监测**: 每更新一个事件或定期进行风险预测
- **风险趋势分析**: 识别风险上升、降低或新发风险
- **死亡率预测**: 基于当前状态预测患者死亡率
- **出院时间预测**: 预测患者预期出院时间
- **风险预警**: 及时向医生提供风险预警信息

#### 2.1.2 诊疗建议功能
- **主动问答交互**: 医生可主动询问诊疗建议
- **基于当前状态**: 基于患者最新状态提供建议
- **多维度建议**: 涵盖诊断、治疗、护理、用药等方面
- **循证医学支持**: 基于医学证据和指南提供建议

### 2.2 技术挑战

#### 2.2.1 数据规模挑战
- **事件数量庞大**: 单个患者可能有数千个事件
- **实时性要求**: 需要实时处理新事件
- **上下文限制**: 无法将所有历史事件放入LLM上下文

#### 2.2.2 专业性挑战
- **医疗知识要求**: 需要深厚的医学专业知识
- **准确性要求**: 医疗建议必须高度准确
- **安全性要求**: 避免错误的医疗建议

## 3. 系统架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ICU多Agent系统                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 风险预测Agent│  │ 诊疗建议Agent│  │ 记忆管理Agent│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 事件处理器  │  │ 知识库管理  │  │ 向量数据库  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ LLM客户端   │  │ 医疗知识库  │  │ 风险评估引擎│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Agent角色设计

#### 3.2.1 风险预测Agent
- **职责**: 实时监测患者风险状态
- **输入**: 患者事件序列、生命体征、实验室检查等
- **输出**: 风险评分、风险趋势、死亡率预测、出院时间预测
- **触发**: 新事件到达或定时触发

#### 3.2.2 诊疗建议Agent
- **职责**: 提供专业诊疗建议
- **输入**: 医生查询、患者当前状态、相关医学知识
- **输出**: 诊疗建议、用药建议、护理建议
- **触发**: 医生主动查询

#### 3.2.3 记忆管理Agent
- **职责**: 管理患者历史信息
- **功能**: 事件压缩、重要信息提取、上下文管理
- **策略**: 分层记忆、重要性排序、时间衰减

### 3.3 核心组件设计

#### 3.3.1 事件处理器
```python
class EventProcessor:
    def __init__(self):
        self.event_parsers = {
            'surgery': SurgeryEventParser(),
            'order': OrderEventParser(),
            'lab': LabEventParser(),
            'nursing': NursingEventParser(),
            'history': HistoryEventParser()
        }
    
    def process_event(self, event):
        """处理单个事件，提取关键信息"""
        parser = self.event_parsers.get(event['event_type'])
        return parser.parse(event)
    
    def extract_risks(self, event):
        """从事件中提取风险信息"""
        # 基于事件内容和风险分类体系提取风险
        pass
```

#### 3.3.2 记忆管理系统
```python
class MemoryManager:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.summary_chain = SummaryChain()
    
    def add_event(self, patient_id, event):
        """添加新事件到记忆系统"""
        # 1. 向量化事件
        # 2. 存储到向量数据库
        # 3. 更新摘要
        pass
    
    def get_relevant_context(self, patient_id, query, limit=10):
        """获取相关上下文信息"""
        # 基于查询检索相关事件
        pass
    
    def generate_summary(self, patient_id, time_window):
        """生成时间窗口内的摘要"""
        pass
```

#### 3.3.3 风险评估引擎
```python
class RiskAssessmentEngine:
    def __init__(self):
        self.risk_categories = self.load_risk_categories()
        self.risk_models = self.load_risk_models()
    
    def assess_risks(self, patient_state):
        """评估患者当前风险状态"""
        risks = {}
        for category, risks_list in self.risk_categories.items():
            for risk in risks_list:
                score = self.calculate_risk_score(risk, patient_state)
                if score > threshold:
                    risks[risk] = score
        return risks
    
    def predict_mortality(self, patient_state):
        """预测死亡率"""
        # 基于APACHE II、SOFA等评分系统
        pass
    
    def predict_discharge_time(self, patient_state):
        """预测出院时间"""
        pass
```

## 4. 实现方案

### 4.1 数据预处理和Bench构建

#### 4.1.1 数据清洗
```python
def clean_patient_data(patient_data):
    """清洗患者数据"""
    # 1. 移除错误事件
    # 2. 标准化时间格式
    # 3. 提取关键信息
    # 4. 构建事件向量
    pass
```

#### 4.1.2 风险标注
```python
def annotate_risks(events, risk_table):
    """为事件标注风险"""
    for event in events:
        event['risks'] = extract_risks_from_event(event, risk_table)
    return events
```

#### 4.1.3 Bench构建
```python
def build_icu_bench():
    """构建ICU测试基准"""
    # 1. 选择代表性患者
    # 2. 构建测试场景
    # 3. 定义评估指标
    # 4. 创建测试用例
    pass
```

### 4.2 记忆系统实现

#### 4.2.1 分层记忆策略
```python
class HierarchicalMemory:
    def __init__(self):
        self.short_term = ShortTermMemory()  # 最近事件
        self.medium_term = MediumTermMemory()  # 重要事件
        self.long_term = LongTermMemory()  # 摘要信息
    
    def add_event(self, event):
        """添加事件到相应层级"""
        # 根据重要性分配到不同层级
        pass
    
    def get_context(self, query):
        """获取相关上下文"""
        # 从各层级检索相关信息
        pass
```

#### 4.2.2 事件重要性评估
```python
def calculate_event_importance(event):
    """计算事件重要性"""
    factors = {
        'event_type_weight': get_event_type_weight(event['event_type']),
        'content_complexity': calculate_content_complexity(event['event_content']),
        'time_recency': calculate_time_recency(event['timestamp']),
        'risk_indicator': len(event.get('risks', [])),
        'medical_impact': assess_medical_impact(event)
    }
    return weighted_sum(factors)
```

### 4.3 风险预测实现

#### 4.3.1 实时风险监测
```python
class RiskMonitor:
    def __init__(self):
        self.risk_engine = RiskAssessmentEngine()
        self.alert_system = AlertSystem()
    
    def monitor_patient(self, patient_id):
        """实时监测患者风险"""
        # 1. 获取最新状态
        # 2. 评估风险
        # 3. 检测变化
        # 4. 发送预警
        pass
    
    def detect_risk_changes(self, current_risks, previous_risks):
        """检测风险变化"""
        changes = {
            'increased': [],
            'decreased': [],
            'new': [],
            'resolved': []
        }
        # 比较风险变化
        return changes
```

#### 4.3.2 预测模型
```python
class PredictionModels:
    def __init__(self):
        self.mortality_model = MortalityPredictionModel()
        self.discharge_model = DischargePredictionModel()
        self.risk_trend_model = RiskTrendModel()
    
    def predict_mortality(self, patient_state):
        """预测死亡率"""
        # 基于APACHE II、SOFA、GCS等评分
        apache_score = calculate_apache_score(patient_state)
        sofa_score = calculate_sofa_score(patient_state)
        return self.mortality_model.predict(apache_score, sofa_score)
    
    def predict_discharge(self, patient_state):
        """预测出院时间"""
        # 基于当前状态和恢复趋势
        pass
```

### 4.4 诊疗建议实现

#### 4.4.1 知识库构建
```python
class MedicalKnowledgeBase:
    def __init__(self):
        self.guidelines = load_clinical_guidelines()
        self.drug_database = load_drug_database()
        self.procedure_database = load_procedure_database()
    
    def search_relevant_knowledge(self, query, patient_state):
        """搜索相关知识"""
        # 基于查询和患者状态检索相关知识
        pass
```

#### 4.4.2 建议生成
```python
class TreatmentAdvisor:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.llm_client = LLMClient()
    
    def generate_advice(self, query, patient_context):
        """生成诊疗建议"""
        # 1. 理解查询意图
        # 2. 检索相关知识
        # 3. 分析患者状态
        # 4. 生成建议
        # 5. 验证建议合理性
        pass
```

## 5. 技术实现细节

### 5.1 向量数据库设计

#### 5.1.1 事件向量化
```python
def vectorize_event(event):
    """将事件转换为向量"""
    # 1. 提取文本内容
    # 2. 使用医疗领域预训练模型编码
    # 3. 添加元数据特征
    # 4. 生成向量表示
    pass
```

#### 5.1.2 相似性搜索
```python
def find_similar_events(query, patient_events, top_k=10):
    """查找相似事件"""
    # 使用向量相似性搜索
    pass
```

### 5.2 LLM集成

#### 5.2.1 提示工程
```python
class PromptBuilder:
    def __init__(self):
        self.templates = load_prompt_templates()
    
    def build_risk_assessment_prompt(self, patient_context):
        """构建风险评估提示"""
        template = self.templates['risk_assessment']
        return template.format(
            patient_context=patient_context,
            risk_categories=self.risk_categories
        )
    
    def build_treatment_advice_prompt(self, query, patient_context):
        """构建诊疗建议提示"""
        template = self.templates['treatment_advice']
        return template.format(
            query=query,
            patient_context=patient_context
        )
```

#### 5.2.2 输出解析
```python
def parse_llm_response(response, task_type):
    """解析LLM响应"""
    if task_type == 'risk_assessment':
        return parse_risk_assessment(response)
    elif task_type == 'treatment_advice':
        return parse_treatment_advice(response)
    else:
        return response
```

### 5.3 系统集成

#### 5.3.1 事件流处理
```python
class EventStreamProcessor:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.risk_monitor = RiskMonitor()
        self.treatment_advisor = TreatmentAdvisor()
    
    def process_new_event(self, patient_id, event):
        """处理新事件"""
        # 1. 更新记忆系统
        self.memory_manager.add_event(patient_id, event)
        
        # 2. 触发风险评估
        risks = self.risk_monitor.assess_risks(patient_id)
        
        # 3. 发送预警（如有必要）
        if self.should_alert(risks):
            self.send_alert(patient_id, risks)
    
    def handle_doctor_query(self, patient_id, query):
        """处理医生查询"""
        # 1. 获取患者上下文
        context = self.memory_manager.get_context(patient_id, query)
        
        # 2. 生成诊疗建议
        advice = self.treatment_advisor.generate_advice(query, context)
        
        return advice
```

## 6. 评估和测试

### 6.1 评估指标

#### 6.1.1 风险预测评估
- **准确性**: 风险预测的准确性
- **及时性**: 风险检测的及时性
- **敏感性**: 对真实风险的敏感性
- **特异性**: 对非风险的识别能力

#### 6.1.2 诊疗建议评估
- **相关性**: 建议与查询的相关性
- **准确性**: 医学建议的准确性
- **实用性**: 建议的实用性
- **安全性**: 建议的安全性

### 6.2 测试场景

#### 6.2.1 风险预测测试
```python
def test_risk_prediction():
    """测试风险预测功能"""
    # 1. 使用历史数据测试
    # 2. 验证预测准确性
    # 3. 测试实时性能
    pass
```

#### 6.2.2 诊疗建议测试
```python
def test_treatment_advice():
    """测试诊疗建议功能"""
    # 1. 模拟医生查询
    # 2. 验证建议质量
    # 3. 测试响应时间
    pass
```

## 7. 部署和运维

### 7.1 系统部署
- **容器化部署**: 使用Docker容器化各个组件
- **微服务架构**: 将不同功能模块独立部署
- **负载均衡**: 处理多患者并发访问
- **监控告警**: 实时监控系统状态

### 7.2 数据安全
- **数据加密**: 患者数据加密存储
- **访问控制**: 严格的权限管理
- **审计日志**: 完整的操作审计
- **合规性**: 符合医疗数据保护法规

## 8. 总结

本ICU多Agent系统设计充分考虑了ICU环境的特殊性和复杂性，通过分层记忆管理、实时风险监测和专业诊疗建议，为ICU医生提供智能化的决策支持。系统的核心优势在于：

1. **鲁棒的记忆系统**: 能够处理大量历史事件数据
2. **专业的医疗知识**: 集成丰富的医学知识库
3. **实时响应能力**: 支持实时事件处理和风险预警
4. **可扩展的架构**: 支持功能模块的独立扩展

通过这个系统，ICU医生可以获得更全面的患者信息，提前识别风险，获得专业的诊疗建议，从而提高医疗质量和患者安全性。
