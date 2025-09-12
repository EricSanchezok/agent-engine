# Agent Engine 打包指南

## 概述

本项目已经配置好了 `agent_engine` 库的打包环境，可以将其打包成 pip 包并上传到 PyPI。

## 项目结构

```
agent-engine/
├── agent_engine/          # 核心库代码（会被打包）
│   ├── a2a_client/
│   ├── agent/
│   ├── agent_logger/
│   ├── llm_client/
│   ├── memory/
│   ├── prompt/
│   └── utils/
├── agents/                # 基于库的应用代码（不打包）
├── core/                  # 项目特定代码（不打包）
├── test/                  # 测试代码（不打包）
├── service/               # 服务代码（不打包）
├── scripts/               # 脚本文件（不打包）
├── web/                   # Web界面（不打包）
├── docs/                  # 文档（不打包）
├── database/              # 数据库文件（不打包）
├── logs/                  # 日志文件（不打包）
├── pyproject.toml         # 项目配置
├── MANIFEST.in           # 打包文件控制
├── build_and_upload.py   # Python打包脚本
└── build.bat            # Windows批处理脚本
```

## 打包配置

### 1. pyproject.toml 配置

- **dependencies**: agent_engine 库的核心依赖
- **opts**: 开发和项目特定的依赖（不包含在包中）
- **包发现规则**: 只包含 `agent_engine*` 包
- **排除规则**: 排除所有项目特定的目录和文件

### 2. MANIFEST.in 配置

- 明确指定包含和排除的文件
- 确保只打包必要的文件
- 排除开发环境和项目特定文件

## 使用方法

### 方法一：使用批处理脚本（推荐）

```bash
# Windows
./build.bat
```

### 方法二：使用Python脚本

```bash
# 构建包
python build_and_upload.py

# 上传到测试PyPI
python build_and_upload.py --test

# 上传到正式PyPI
python build_and_upload.py --upload
```

### 方法三：手动命令

```bash
# 1. 同步依赖
uv sync --extra opts

# 2. 构建包
uv build

# 3. 检查包
twine check dist/*

# 4. 上传到测试PyPI
twine upload --repository testpypi dist/*

# 5. 上传到正式PyPI
twine upload dist/*
```

## 构建产物

构建完成后，`dist/` 目录会包含：

- `agent_engine-0.1.0-py3-none-any.whl` - 轮子包（推荐）
- `agent_engine-0.1.0.tar.gz` - 源码包

## 包内容

打包的 `agent_engine` 库包含：

- 所有 Python 模块（.py 文件）
- 配置文件（.yaml, .yml 文件）
- 示例文件（.example.* 文件）
- README.md 和 pyproject.toml

## 安装测试

构建完成后，可以本地测试安装：

```bash
pip install dist/agent_engine-0.1.0-py3-none-any.whl
```

## 上传到PyPI

### 1. 注册PyPI账号

- 访问 [PyPI](https://pypi.org) 注册账号
- 访问 [Test PyPI](https://test.pypi.org) 注册测试账号

### 2. 配置认证

```bash
# 安装twine（如果还没有）
uv add --dev twine

# 配置PyPI认证
twine upload --repository testpypi dist/*
# 输入用户名和密码

# 配置正式PyPI认证
twine upload dist/*
# 输入用户名和密码
```

### 3. 上传流程

1. 先上传到 Test PyPI 测试
2. 确认无误后上传到正式 PyPI

## 版本管理

更新版本时，修改 `pyproject.toml` 中的 `version` 字段：

```toml
version = "0.1.1"  # 更新版本号
```

## 注意事项

1. **依赖管理**: 核心依赖放在 `dependencies` 中，开发依赖放在 `opts` 中
2. **文件排除**: 确保项目特定文件不会被包含在包中
3. **版本控制**: 每次发布前更新版本号
4. **测试**: 上传前先在本地测试安装
5. **文档**: 确保 README.md 包含足够的使用说明

## 故障排除

### 常见问题

1. **构建失败**: 检查 `pyproject.toml` 语法
2. **包过大**: 检查 `MANIFEST.in` 排除规则
3. **依赖问题**: 确认所有依赖都在 `dependencies` 中
4. **上传失败**: 检查 PyPI 认证和网络连接

### 清理构建文件

```bash
# 清理构建目录
rm -rf build/ dist/ agent_engine.egg-info/
```

## 自动化建议

可以考虑设置 GitHub Actions 来自动化：

1. 代码推送时自动构建
2. 打标签时自动发布到 PyPI
3. 自动运行测试和检查

这样就能实现完全自动化的包发布流程。
