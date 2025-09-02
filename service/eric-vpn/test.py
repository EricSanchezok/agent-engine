import requests

# 定义要测试的目标URL
target_url = "http://10.244.9.104:9900/.well-known/agent-card.json"

print(f"正在尝试访问: {target_url}")

try:
    # 发送GET请求，设置一个超时时间（例如5秒）
    response = requests.get(target_url, timeout=5)

    # 检查响应状态码
    if response.status_code == 200:
        print(f"成功访问！状态码: {response.status_code}")
        print("响应内容 (前150个字符):")
        print(response.text[:150])
    else:
        print(f"访问成功，但状态码不是200。状态码: {response.status_code}")
        print("响应内容:")
        print(response.text)

except requests.exceptions.ConnectionError as e:
    print(f"访问失败: 连接错误。")
    print(f"请检查目标IP地址和端口是否正确，以及网络连接是否通畅。")
    print(f"详细错误: {e}")

except requests.exceptions.Timeout as e:
    print(f"访问失败: 请求超时。")
    print(f"目标服务器可能响应缓慢或无法访问。")
    print(f"详细错误: {e}")

except requests.exceptions.RequestException as e:
    print(f"发生了未知请求错误。")
    print(f"详细错误: {e}")