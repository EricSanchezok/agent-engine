#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICU Agent Backend API Server
Provides REST API endpoints for the ICU Agent frontend application.
"""

import time
import random
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Sample patient data
PATIENTS_DATA = {
    1: {
        "id": 1,
        "name": "张某某",
        "bed_number": "A-101",
        "age": 65,
        "gender": "male",
        "condition": "急性心肌梗死",
        "admission_date": "2024-01-15",
        "vital_signs": {
            "heart_rate": 98,
            "blood_pressure": "145/90",
            "temperature": 37.2,
            "respiratory_rate": 18,
            "oxygen_saturation": 94
        },
        "lab_results": {
            "creatinine": 1.8,
            "troponin": 25.3,
            "bnp": 1850,
            "lactate": 2.1
        },
        "medications": [
            "阿司匹林 100mg qd",
            "美托洛尔 25mg bid",
            "阿托伐他汀 20mg qn"
        ],
        "risk_factors": ["高血压", "糖尿病", "吸烟史"]
    },
    2: {
        "id": 2,
        "name": "李某某",
        "bed_number": "A-102",
        "age": 72,
        "gender": "female",
        "condition": "急性呼吸窘迫综合征",
        "admission_date": "2024-01-14",
        "vital_signs": {
            "heart_rate": 115,
            "blood_pressure": "160/95",
            "temperature": 38.5,
            "respiratory_rate": 28,
            "oxygen_saturation": 88
        },
        "lab_results": {
            "creatinine": 2.3,
            "procalcitonin": 8.5,
            "d_dimer": 2100,
            "lactate": 3.2
        },
        "medications": [
            "呋塞米 40mg bid",
            "去甲肾上腺素 0.1μg/kg/min",
            "头孢曲松 2g qd"
        ],
        "risk_factors": ["慢阻肺", "高血压", "心房颤动"]
    }
}

# Risk prediction models simulation
RISK_MODELS = {
    "acute_kidney_injury": {
        "factors": ["creatinine", "urine_output", "medications"],
        "weights": [0.4, 0.3, 0.3]
    },
    "septic_shock": {
        "factors": ["temperature", "heart_rate", "lactate", "procalcitonin"],
        "weights": [0.25, 0.25, 0.25, 0.25]
    },
    "cardiac_arrhythmia": {
        "factors": ["heart_rate", "troponin", "medications"],
        "weights": [0.4, 0.4, 0.2]
    },
    "respiratory_failure": {
        "factors": ["respiratory_rate", "oxygen_saturation", "temperature"],
        "weights": [0.4, 0.4, 0.2]
    }
}

# Clinical knowledge base for question answering
CLINICAL_KNOWLEDGE = {
    "keywords": {
        "血压": ["高血压", "低血压", "血压管理", "降压药物"],
        "心率": ["心律不齐", "心动过速", "心动过缓", "心律失常"],
        "肾功能": ["肾衰竭", "肌酐", "尿素氮", "肾脏替代治疗"],
        "感染": ["脓毒症", "抗生素", "炎症指标", "感染控制"],
        "呼吸": ["呼吸衰竭", "机械通气", "氧合", "ARDS"],
        "用药": ["药物相互作用", "剂量调整", "不良反应", "禁忌症"]
    },
    "responses": {
        "血压": "根据患者当前血压水平，建议密切监测血压变化。如血压持续升高，可考虑调整降压药物剂量或添加新的降压药物。",
        "心率": "患者心率需要持续监测。如出现心律不齐，建议进行心电图检查并考虑抗心律失常治疗。",
        "肾功能": "患者肌酐水平提示肾功能受损，建议监测尿量、调整肾毒性药物剂量，必要时考虑肾脏替代治疗。",
        "感染": "根据炎症指标，建议进行感染源筛查，合理使用抗生素，注意监测感染指标变化。",
        "呼吸": "患者氧合状况需要关注，建议优化氧疗参数，必要时考虑无创或有创机械通气支持。",
        "用药": "建议审核当前用药方案，注意药物相互作用和剂量调整，特别是肾功能不全患者的用药。"
    }
}

def generate_risk_score(patient_data, risk_type):
    """Generate risk score for specific condition"""
    if risk_type not in RISK_MODELS:
        return random.randint(20, 80)
    
    # Simulate risk calculation based on patient data
    base_risk = random.randint(30, 70)
    
    # Adjust based on patient age
    age_factor = (patient_data.get("age", 60) - 50) / 30 * 20
    
    # Adjust based on vital signs
    if patient_data.get("vital_signs"):
        vitals = patient_data["vital_signs"]
        if vitals.get("heart_rate", 80) > 100:
            base_risk += 10
        if vitals.get("temperature", 36.5) > 38:
            base_risk += 15
    
    # Adjust based on lab results
    if patient_data.get("lab_results"):
        labs = patient_data["lab_results"]
        if labs.get("creatinine", 1.0) > 1.5:
            base_risk += 20
        if labs.get("lactate", 1.0) > 2.0:
            base_risk += 15
    
    final_risk = min(max(base_risk + age_factor, 10), 95)
    return int(final_risk)

def analyze_question(question, patient_data):
    """Analyze user question and generate appropriate response"""
    question_lower = question.lower()
    
    # Find relevant keywords
    relevant_topics = []
    for topic, keywords in CLINICAL_KNOWLEDGE["keywords"].items():
        if any(keyword in question_lower for keyword in keywords) or topic in question_lower:
            relevant_topics.append(topic)
    
    if not relevant_topics:
        # Generic response for unrecognized questions
        return generate_generic_response(patient_data)
    
    # Generate response based on relevant topics
    responses = []
    for topic in relevant_topics[:2]:  # Limit to top 2 topics
        base_response = CLINICAL_KNOWLEDGE["responses"].get(topic, "")
        if base_response:
            # Customize response with patient-specific data
            customized_response = customize_response(base_response, patient_data, topic)
            responses.append(customized_response)
    
    return " ".join(responses)

def customize_response(base_response, patient_data, topic):
    """Customize response with patient-specific data"""
    patient_name = patient_data.get("name", "患者")
    
    # Add patient-specific data points
    if topic == "血压" and patient_data.get("vital_signs"):
        bp = patient_data["vital_signs"].get("blood_pressure", "未知")
        return f"{patient_name}当前血压为{bp}。{base_response}"
    
    elif topic == "心率" and patient_data.get("vital_signs"):
        hr = patient_data["vital_signs"].get("heart_rate", "未知")
        return f"{patient_name}当前心率为{hr}次/分。{base_response}"
    
    elif topic == "肾功能" and patient_data.get("lab_results"):
        creatinine = patient_data["lab_results"].get("creatinine", "未知")
        return f"{patient_name}当前肌酐值为{creatinine}mg/dL。{base_response}"
    
    elif topic == "呼吸" and patient_data.get("vital_signs"):
        rr = patient_data["vital_signs"].get("respiratory_rate", "未知")
        spo2 = patient_data["vital_signs"].get("oxygen_saturation", "未知")
        return f"{patient_name}当前呼吸频率{rr}次/分，血氧饱和度{spo2}%。{base_response}"
    
    return f"关于{patient_name}，{base_response}"

def generate_generic_response(patient_data):
    """Generate generic response when no specific topics are identified"""
    patient_name = patient_data.get("name", "患者")
    condition = patient_data.get("condition", "当前病情")
    
    responses = [
        f"根据{patient_name}的{condition}，我建议密切关注患者的生命体征变化。",
        f"对于{patient_name}的情况，建议定期评估病情进展并调整治疗方案。",
        f"针对{patient_name}的病情，建议多学科团队会诊以制定最佳治疗策略。"
    ]
    
    return random.choice(responses)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from frontend"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        message = data.get('message', '').strip()
        
        if not patient_id or not message:
            return jsonify({'error': 'Patient ID and message are required'}), 400
        
        # Get patient data
        patient_data = PATIENTS_DATA.get(patient_id)
        if not patient_data:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Simulate processing delay
        time.sleep(random.uniform(1.0, 2.5))
        
        # Analyze question and generate response
        response = analyze_question(message, patient_data)
        
        # Add some clinical context
        if "风险" in message or "预测" in message:
            risk_info = generate_risk_assessment(patient_data)
            response += f" {risk_info}"
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def generate_risk_assessment(patient_data):
    """Generate risk assessment for patient"""
    risks = []
    
    # Generate risks for common ICU complications
    risk_types = ["acute_kidney_injury", "septic_shock", "cardiac_arrhythmia", "respiratory_failure"]
    
    for risk_type in risk_types:
        score = generate_risk_score(patient_data, risk_type)
        risk_name = {
            "acute_kidney_injury": "急性肾损伤",
            "septic_shock": "感染性休克",
            "cardiac_arrhythmia": "心律失常",
            "respiratory_failure": "呼吸衰竭"
        }.get(risk_type, risk_type)
        
        risks.append(f"{risk_name}风险为{score}%")
    
    return f"当前风险评估：{', '.join(risks[:2])}。"

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get patient data"""
    patient_data = PATIENTS_DATA.get(patient_id)
    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    return jsonify(patient_data)

@app.route('/api/patient/<int:patient_id>/risks', methods=['GET'])
def get_patient_risks(patient_id):
    """Get patient risk assessment"""
    patient_data = PATIENTS_DATA.get(patient_id)
    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    risks = []
    risk_types = ["acute_kidney_injury", "septic_shock", "cardiac_arrhythmia", "respiratory_failure"]
    
    for risk_type in risk_types:
        score = generate_risk_score(patient_data, risk_type)
        risk_name = {
            "acute_kidney_injury": "急性肾损伤",
            "septic_shock": "感染性休克", 
            "cardiac_arrhythmia": "心律失常",
            "respiratory_failure": "呼吸衰竭"
        }.get(risk_type, risk_type)
        
        level = "high" if score >= 70 else "medium" if score >= 40 else "low"
        
        risks.append({
            "name": risk_name,
            "probability": score,
            "level": level,
            "type": risk_type
        })
    
    return jsonify({"risks": risks})

@app.route('/api/patient/<int:patient_id>/events', methods=['GET'])
def get_patient_events(patient_id):
    """Get patient events timeline"""
    patient_data = PATIENTS_DATA.get(patient_id)
    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Generate sample events
    events = []
    now = datetime.now()
    
    # Generate events for the last 24 hours
    for i in range(8):
        event_time = now - timedelta(hours=i*3, minutes=random.randint(0, 59))
        
        event_types = [
            {
                "title": "血压异常",
                "description": f"收缩压升高至{random.randint(160, 200)}mmHg",
                "type": "warning"
            },
            {
                "title": "用药记录", 
                "description": f"静脉注射{random.choice(['呋塞米', '美托洛尔', '阿司匹林'])}",
                "type": "info"
            },
            {
                "title": "检验结果",
                "description": f"肌酐值{random.choice(['上升', '下降'])}至{random.uniform(1.5, 3.0):.1f}mg/dL",
                "type": "critical" if random.random() < 0.3 else "info"
            },
            {
                "title": "体征监测",
                "description": f"心率{random.randint(80, 120)}次/分，呼吸{random.randint(16, 28)}次/分",
                "type": "normal"
            }
        ]
        
        event = random.choice(event_types)
        events.append({
            "time": event_time.strftime("%H:%M"),
            "title": event["title"],
            "description": event["description"],
            "type": event["type"],
            "timestamp": event_time.isoformat()
        })
    
    # Sort by time (newest first)
    events.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return jsonify({"events": events})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting ICU Agent Backend Server...")
    print("Server will be available at: http://localhost:5000")
    print("API endpoints:")
    print("  POST /api/chat - Chat with ICU Agent")
    print("  GET /api/patient/<id> - Get patient data") 
    print("  GET /api/patient/<id>/risks - Get patient risks")
    print("  GET /api/patient/<id>/events - Get patient events")
    print("  GET /api/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
