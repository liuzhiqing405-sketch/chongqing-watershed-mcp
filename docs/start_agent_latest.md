# 启动当前版本智能体

在 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_agent_latest.ps1
```

入口先检查 WebUI 8780 端口和水文/图件活动任务，再根据服务记录与实时进程身份核对两个 MCP。全部预检完成后才重载已核验的空闲服务；缺失服务由正常启动入口启动。进程创建时间复核防止 PID 复用。原对话、输入和模型文件保持不变。

如果旧 WebUI 仍在运行，先在它的启动终端按 Ctrl+C，再执行该命令。未知端口占用或正在运行的任务会明确阻止重启，不强行结束进程。

MCP 协议检查通过后，在本机掩码输入 DeepSeek API Key，浏览器打开 http://127.0.0.1:8780 。密钥不写入源码、配置、启动参数或会话。结束 WebUI 后 MCP 服务继续运行。

只验证启动条件、不改变进程或输入密钥：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_agent_latest.ps1 -CheckOnly
```

测试 GRASS 新方案时，新建对话，重新画选区、确认范围并选择有有效降雨数据的事件后计算。恢复旧对话仍会显示旧运行的成果，启动服务不会自动替换它。可以先测试知识问答与当前任务地点、质量原因提问，确认答复正常，再提交计算。
