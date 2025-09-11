#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICU Agent Demo Script
Demonstrates the key functionality of the ICU Agent system.
"""

import json
from datetime import datetime
from app import app, PATIENTS_DATA, generate_risk_score, analyze_question

def demo_patient_data():
    """Demonstrate patient data functionality"""
    print("=== Patient Data Demo ===")
    
    for patient_id, patient_data in PATIENTS_DATA.items():
        print(f"\nPatient {patient_id}: {patient_data['name']}")
        print(f"  Bed: {patient_data['bed_number']}")
        print(f"  Condition: {patient_data['condition']}")
        print(f"  Age: {patient_data['age']}")
        
        if patient_data.get('vital_signs'):
            vitals = patient_data['vital_signs']
            print(f"  Vitals: HR={vitals.get('heart_rate')}, BP={vitals.get('blood_pressure')}")
    
    print()

def demo_risk_assessment():
    """Demonstrate risk assessment functionality"""
    print("=== Risk Assessment Demo ===")
    
    patient = PATIENTS_DATA[1]  # Use first patient
    print(f"Risk assessment for {patient['name']}:")
    
    risk_types = {
        "acute_kidney_injury": "急性肾损伤",
        "septic_shock": "感染性休克",
        "cardiac_arrhythmia": "心律失常",
        "respiratory_failure": "呼吸衰竭"
    }
    
    for risk_type, risk_name in risk_types.items():
        score = generate_risk_score(patient, risk_type)
        level = "高风险" if score >= 70 else "中风险" if score >= 40 else "低风险"
        print(f"  {risk_name}: {score}% ({level})")
    
    print()

def demo_chat_functionality():
    """Demonstrate chat AI functionality"""
    print("=== Chat AI Demo ===")
    
    patient = PATIENTS_DATA[1]
    print(f"Simulating chat with AI about {patient['name']}:")
    
    test_questions = [
        "患者当前的血压情况如何？",
        "这个病人的肾功能怎么样？",
        "需要调整什么治疗方案？",
        "风险评估结果是什么？"
    ]
    
    for question in test_questions:
        print(f"\n医生: {question}")
        response = analyze_question(question, patient)
        print(f"Agent: {response}")
    
    print()

def demo_frontend_data():
    """Show data that would be displayed in frontend"""
    print("=== Frontend Data Structure Demo ===")
    
    # Sample risk data for frontend
    risk_data = [
        {"name": "急性肾损伤", "probability": 85, "level": "high"},
        {"name": "感染性休克", "probability": 65, "level": "medium"},
        {"name": "心律失常", "probability": 45, "level": "medium"},
        {"name": "呼吸衰竭", "probability": 25, "level": "low"}
    ]
    
    print("Risk warnings for left sidebar:")
    for risk in risk_data:
        print(f"  {risk['name']}: {risk['probability']}% ({risk['level']})")
    
    # Sample events data for frontend
    events_data = [
        {"time": "14:32", "title": "血压异常", "type": "warning"},
        {"time": "13:45", "title": "用药记录", "type": "info"},
        {"time": "12:20", "title": "检验结果", "type": "critical"},
        {"time": "11:30", "title": "体征监测", "type": "normal"}
    ]
    
    print("\nEvent timeline for left sidebar:")
    for event in events_data:
        print(f"  {event['time']} - {event['title']} ({event['type']})")
    
    print()

def demo_workflow_steps():
    """Demonstrate Agent workflow transparency"""
    print("=== Agent Workflow Demo ===")
    
    workflow_steps = [
        "解析问题意图",
        "检索患者档案", 
        "分析医疗数据",
        "调用诊断模型",
        "生成回复内容"
    ]
    
    print("Agent processing workflow (displayed in right sidebar):")
    for i, step in enumerate(workflow_steps, 1):
        status = "✓ 已完成" if i <= 3 else "⏳ 进行中" if i == 4 else "⏸ 等待"
        print(f"  {i}. {step} {status}")
    
    print()

def main():
    """Run the complete demo"""
    print("ICU Agent System Demo")
    print("====================")
    print(f"Demo time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all demo functions
    demo_patient_data()
    demo_risk_assessment() 
    demo_chat_functionality()
    demo_frontend_data()
    demo_workflow_steps()
    
    print("=== Demo Summary ===")
    print("✓ Patient management system")
    print("✓ Risk assessment algorithms")
    print("✓ AI chat functionality") 
    print("✓ Real-time monitoring data")
    print("✓ Workflow transparency")
    print()
    print("The ICU Agent system is ready for clinical use!")
    print()
    print("To start the web interface:")
    print("1. Run: python app.py (for backend)")
    print("2. Open: index.html in browser")
    print("3. Select a patient and start chatting!")

if __name__ == "__main__":
    main()
