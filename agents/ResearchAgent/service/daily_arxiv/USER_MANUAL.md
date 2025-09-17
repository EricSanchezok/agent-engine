# Daily arXiv 论文报告 API - 用户手册

## 概述

本API服务提供对自动化研究系统生成的每日arXiv论文分析报告的访问。该服务允许用户在指定的日期范围内查询论文报告，并获取详细的markdown格式分析报告。

## 基础URL

API服务器默认运行在5000端口。访问URL：
- 本地：`http://localhost:5000`
- 网络：`http://[服务器IP]:5000`

## API 接口

### 1. 获取论文报告

**接口地址：** `POST /api/paper-reports`

**功能描述：** 获取指定日期范围内的论文分析报告。

**请求体：**
```json
{
  "date_range": {
    "start_date": "20250916",
    "end_date": "20250917"
  }
}
```

**请求参数：**
- `start_date` (字符串，必需)：开始日期，格式为YYYYMMDD（包含该日期）
- `end_date` (字符串，必需)：结束日期，格式为YYYYMMDD（包含该日期）

**日期范围行为：**
API使用**闭区间**（包含边界），意味着start_date和end_date都包含在结果中。例如：
- `start_date: "20250914", end_date: "20250916"` 返回2025-09-14、2025-09-15和2025-09-16的论文
- `start_date: "20250914", end_date: "20250914"` 仅返回2025-09-14的论文

**响应格式：**
```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "title": "论文标题",
      "authors": ["作者1", "作者2"],
      "categories": ["cs.AI", "cs.LG"],
      "timestamp": "20250916T0000",
      "pdf_url": "https://arxiv.org/pdf/2509.13311v1.pdf",
      "report": "### 论文标题\n**信号源:** 机构\n\n**一句话总结**\n[总结内容...]\n\n**核心创新**\n[创新详情...]\n\n**技术亮点**\n[技术详情...]\n\n**研究影响**\n[影响分析...]\n\n**关键洞察**\n[关键洞察...]"
    }
  ]
}
```

**响应字段：**
- `code` (整数)：状态码（0 = 成功，1 = 错误）
- `message` (字符串)：状态消息或错误描述
- `data` (数组)：论文报告对象数组
  - `title` (字符串)：论文标题（从元数据中提取）
  - `authors` (数组)：作者列表（从arXiv元数据中提取）
  - `categories` (数组)：arXiv分类（从arXiv元数据中提取）
  - `timestamp` (字符串)：处理时间戳，格式为YYYYMMDDTHHMM
  - `pdf_url` (字符串)：arXiv PDF的直接链接（从元数据中提取）
  - `report` (字符串)：完整的markdown格式分析报告

### 2. 健康检查

**接口地址：** `GET /api/health`

**功能描述：** 检查API服务器是否正在运行。

**响应：**
```json
{
  "code": 0,
  "message": "API server is running",
  "data": {
    "status": "healthy",
    "timestamp": "2025-01-16T10:30:00.123456"
  }
}
```

### 3. 处理状态

**接口地址：** `GET /api/status`

**功能描述：** 获取每日arXiv系统的当前处理状态。

**响应：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "2025-01-16": {
      "status": "completed",
      "started_at": "2025-01-16T08:00:00",
      "completed_at": "2025-01-16T08:45:00",
      "result_file": "daily_result_2025-01-16.json"
    }
  }
}
```

## 错误处理

### 常见错误响应

**无效日期格式：**
```json
{
  "code": 1,
  "message": "Invalid date format: time data '2025-13-45' does not match format '%Y%m%d'. Expected format: YYYYMMDD",
  "data": []
}
```

**无效日期范围：**
```json
{
  "code": 0,
  "message": "Invalid date range: start_date cannot be later than end_date",
  "data": []
}
```

**未找到报告：**
```json
{
  "code": 0,
  "message": "No paper reports found for date range 20250916 to 20250917. This could be because: 1) No papers were processed for these dates, 2) The daily arXiv processing has not run yet, or 3) The reports are still being generated.",
  "data": []
}
```

**缺少请求体：**
```json
{
  "code": 1,
  "message": "Request body is required",
  "data": []
}
```

## 使用示例

### Python 示例

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/api/paper-reports"

# 示例1：获取多天日期范围内的论文
data_range = {
    "date_range": {
        "start_date": "20250914",
        "end_date": "20250916"
    }
}

# 示例2：获取单天论文（闭区间）
data_single = {
    "date_range": {
        "start_date": "20250914",
        "end_date": "20250914"  # 相同日期用于单天查询
    }
}

# 发送请求
response = requests.post(url, json=data_single)

# 检查响应
if response.status_code == 200:
    result = response.json()
    
    if result["code"] == 0:
        print(f"成功: {result['message']}")
        print(f"找到 {len(result['data'])} 篇论文")
        
        for paper in result["data"]:
            print(f"\n标题: {paper['title']}")
            print(f"作者: {', '.join(paper['authors'])}")
            print(f"分类: {', '.join(paper['categories'])}")
            print(f"PDF链接: {paper['pdf_url']}")
            print(f"报告预览: {paper['report'][:200]}...")
    else:
        print(f"错误: {result['message']}")
else:
    print(f"HTTP错误: {response.status_code}")
```

### cURL 示例

```bash
# 示例1：获取多天日期范围内的论文
curl -X POST http://localhost:5000/api/paper-reports \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "20250914",
      "end_date": "20250916"
    }
  }'

# 示例2：获取单天论文（闭区间）
curl -X POST http://localhost:5000/api/paper-reports \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "20250914",
      "end_date": "20250914"
    }
  }'

# 格式化输出响应
curl -X POST http://localhost:5000/api/paper-reports \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "20250914",
      "end_date": "20250914"
    }
  }' | python -m json.tool

# 健康检查
curl -X GET http://localhost:5000/api/health

# 状态检查
curl -X GET http://localhost:5000/api/status
```

### JavaScript/Node.js 示例

```javascript
const axios = require('axios');

async function getPaperReports(startDate, endDate) {
    try {
        const response = await axios.post('http://localhost:5000/api/paper-reports', {
            date_range: {
                start_date: startDate,
                end_date: endDate
            }
        });
        
        const result = response.data;
        
        if (result.code === 0) {
            console.log(`成功: ${result.message}`);
            console.log(`找到 ${result.data.length} 篇论文`);
            
            result.data.forEach(paper => {
                console.log(`\n标题: ${paper.title}`);
                console.log(`作者: ${paper.authors.join(', ')}`);
                console.log(`分类: ${paper.categories.join(', ')}`);
                console.log(`PDF链接: ${paper.pdf_url}`);
            });
        } else {
            console.log(`错误: ${result.message}`);
        }
    } catch (error) {
        console.error('请求失败:', error.message);
    }
}

// 使用示例
getPaperReports('20250916', '20250917');
```

## 报告内容结构

`report`字段包含markdown格式的分析

## 注意事项

1. **日期格式**：所有日期必须使用YYYYMMDD格式（例如，"20250916"表示2025年9月16日）。

2. **日期范围**：API使用**闭区间**（包含边界）。start_date和end_date都包含在结果中。要获取单天数据，请将start_date和end_date设置为相同值。

3. **报告可用性**：报告仅在每日arXiv处理成功完成的日期可用。

4. **处理时间**：每日处理通常在早上运行，可能需要30-60分钟完成。

5. **速率限制**：目前没有实施速率限制，但请负责任地使用API。

6. **数据新鲜度**：报告每日生成，当前日期的报告可能不会立即可用。

## 技术支持

如需API的技术支持或有问题，请联系系统管理员或查看服务器日志获取详细错误信息。
