# Streamlit 后台部署指南

## 当前状态
- Streamlit 已配置监听 `0.0.0.0:8501`
- 反向代理已由他人配置完成

## 后台运行方式

### 方式1：使用后台启动脚本（推荐，简单）

```bash
cd /root/project/agentsociety-ecosim/economist
./run_streamlit_background.sh
```

**优点**：
- 简单快速
- 日志自动保存到 `logs/` 目录
- 无需 root 权限配置

**查看日志**：
```bash
tail -f logs/streamlit_*.log
```

**停止服务**：
```bash
pkill -f "streamlit run streamlit_app.py"
```

### 方式2：使用 systemd 服务（推荐，专业）

**安装服务**：
```bash
sudo cp /root/project/agentsociety-ecosim/economist/streamlit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable streamlit.service
sudo systemctl start streamlit.service
```

**管理服务**：
```bash
# 启动
sudo systemctl start streamlit

# 停止
sudo systemctl stop streamlit

# 重启
sudo systemctl restart streamlit

# 查看状态
sudo systemctl status streamlit

# 查看日志
sudo journalctl -u streamlit -f
```

**优点**：
- 自动重启（崩溃后自动恢复）
- 开机自启
- 系统级管理
- 日志统一管理

### 方式3：使用 nohup（临时方案）

```bash
cd /root/project/agentsociety-ecosim/economist
nohup ./run_streamlit.sh > streamlit.log 2>&1 &
```

## 验证服务运行

```bash
# 检查进程
ps aux | grep streamlit

# 检查端口
netstat -tlnp | grep 8501
# 或
ss -tlnp | grep 8501

# 测试访问（内网）
curl http://localhost:8501
```

## 注意事项

1. **环境变量**：确保 `.env` 文件已配置 API keys
2. **Conda 环境**：确保 `ecosim` 环境已激活
3. **端口冲突**：确保 8501 端口未被占用
4. **日志管理**：定期清理 `logs/` 目录中的旧日志文件
