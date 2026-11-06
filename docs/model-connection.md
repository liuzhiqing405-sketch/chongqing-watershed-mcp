# 在WebUI中连接不同模型

点击右上角“模型未连接 · 设置”或当前模型名，打开连接面板。

本项目的默认接入方式：DeepSeek走官方直连，其余GPT、Claude、Grok、Gemini、Qwen均走4Router。选择对应快捷按钮会设置地址与协议，不会自动应用连接；相同地址可沿用内存密钥，切换到不同地址不会转发旧密钥。不同4Router分组可能需要不同密钥，模型和权限以账户实际返回为准。

| 模型服务 | 默认地址 | 默认协议 | 专属选项 |
| --- | --- | --- | --- |
| DeepSeek | `https://api.deepseek.com` | Chat Completions | 原有thinking开关、effort |
| GPT | `https://4router.net/v1` | Responses | 模型支持的effort |
| Claude | `https://4router.net/v1` | Anthropic Messages | 思考模式、固定预算或effort |
| Grok | `https://4router.net/v1` | Chat Completions，可选Responses | 按版本支持low/medium/high/xhigh；默认不指定 |
| Gemini | `https://4router.net/v1` | Chat Completions；另支持GenerateContent | 思考等级或预算，二选一 |
| Qwen | `https://4router.net/v1` | Chat Completions，默认流式 | enable_thinking、thinking_budget或模型支持的effort |

Grok、Gemini、Qwen适配不代表4Router一定开放某个模型、某种参数或原生入口。公开分组页目前列出Grok与GeminiApi，未明确列出Qwen；不据此断言Qwen不可用，也不往返回列表添加虚构模型。Qwen需要账户实际可用的模型ID与通道。

1. 填入API根地址。4Router的GPT模型点击“4Router GPT”，协议为Responses；Claude模型点击“4Router Claude”，协议为Anthropic Messages。两者均可使用`https://4router.net/v1`，Claude也支持站点根地址，程序只补一次`/v1`。其他兼容服务按其文档选择协议。
2. 在本机密码框输入密钥。地址不变时可留空沿用当前内存密钥；切换地址不会把旧密钥发送到新地址。
3. 点击“读取模型”，从明确显示的候选下拉框选择，也可手动填写model ID。当前填写名称不在返回列表中时会提示；4Router的Claude/GPT名称在确认选择时会匹配对应协议。选择只更新表单，不会自动应用或切换模型。候选列表不替代账户模型权限检查。
4. 设置推理强度。默认表示不额外发送强度参数；可读取到供应商模型元数据时，按所选模型列出候选强度。DeepSeek官方连接另外支持思考开关，显式强度会同步开关：none关闭，其他强度开启。
5. 选择温度、生成token上限、超时和JSON格式。推理模型如不支持温度，可选“服务默认，不发送温度”。思考和最终回答可能共用生成上限。
6. 点击“测试连接”，先发送简短JSON请求，再用真实路由器检查独立问候和工程原理提问。三项均通过才显示“连接和基础意图检查通过”；规则回退不算模型理解通过。检查使用不保存的临时会话，不启动工具、不写入会话历史、不自动应用设置。检查失败会区分请求或响应处理异常、JSON/意图格式错误、意图判断不符；请求异常不能据此判定模型理解能力。接口返回的模型名与填写ID不一致时，两者同时显示。基础意图检查不替代完整任务验证。接口不支持某个参数时会报告失败，不静默降级。

Claude思考模式与effort分别设置：默认不发送额外选项；较新模型可选adaptive，再按模型支持选择low/medium/high/xhigh/max；固定预算enabled主要用于旧模型，预算至少1024且小于生成上限。开启思考时请选择默认温度，不额外发送温度；是否允许关闭或使用某种思考模式以具体模型和中转通道为准。Claude不会收到GPT的reasoning_effort、response_format，DeepSeek专用思考设置也不会透传到Claude。Messages中system单独传递，读取text块作为最终回答，忽略thinking块；未结束的事件流、输出截断或未经授权的工具请求不会被当作成功回答。

Claude的“JSON格式”通过提示词约束，并继续使用原有JSON/意图验证，不冒充原生结构化输出保证；本次没有新增工具调用权限。4Router的Claude令牌分组必须允许第三方程序接入（如其文档列出的ClaudeApi、ClaudeApiV、ClaudeExtern等）；读取列表成功不代表该分组允许任意客户端。

Qwen开启思考时只读取正文content，忽略reasoning_content；流式输出必须收到finish_reason与终止标志，单独收到JSON片段不算成功。REST请求将enable_thinking和thinking_budget放在顶层，不套SDK的extra_body包装。部分旧模型在思考模式下不能保证JSON Object约束，仍保留现有JSON与意图验证，不把思考内容当答案。

Gemini原生GenerateContent将系统提示放入systemInstruction，用户输入放入contents；生成和思考选项放入generationConfig。只接收正常STOP的正文，忽略thought内容，拒绝截断、拦截与未经授权的functionCall。4Router地址填根地址、/v1或/v1beta时，原生生成请求解析到/v1beta/models/模型:generateContent，模型列表仍从4Router的/v1/models读取。该原生入口是否开放需实际验证；默认优先使用兼容Chat入口，不自动失败降级或改换模型。

Grok不能把“默认不指定”理解为关闭推理。不同版本的effort范围不同，旧grok-4或非推理型号不自动附加effort；只有用户显式选择的参数才发送。所有新模型均继续只提供结构化意图和受约束回答，工具执行仍由原状态机和确认步骤控制。

应用后，后续路由、理解、问答、受控改写共用新客户端，原有工具与人工确认门槛保持不变。正在处理请求时拒绝切换；原生水文任务仍按原服务执行。右上角初始显示配置中的模型名，正常回复后优先显示接口返回的model字段；这不是对中转站内部真实权重的独立验证。

连接配置和密钥仅保留在当前服务进程内存，不写YAML、日常会话或浏览器存储。重启后需要重新配置；不要在聊天里提供密钥。

原有终端启动方式仍可用。也可先启动WebUI，再从面板配置：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_agent_webui.ps1 -ConfigureInWeb
```

启动器支持`-BaseUrl`、`-Model`、`-Provider`和`-ConfigPath`，也尊重相应的`SLOPE_AGENT_*`环境变量，不再无条件写回DeepSeek地址和模型名。未指定时仍兼容原DeepSeek配置。

原DeepSeek预设为`deepseek-flash`、`thinking.type=disabled`、温度0、生成上限4096；历史测试必须按实际请求配置描述，不能因新界面新增了高强度选项就重新解释旧成绩。

参考：

- [4Router配置说明](https://docs.4router.net/guides/cc-switch/)：GPT接口使用Responses及模型/推理设置。
- [4Router Claude配置](https://docs.4router.net/guides/cherry-studio/)和[分组说明](https://docs.4router.net/guides/model-groups/)：Claude使用Anthropic协议，第三方接入受分组限制。
- [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create)、[思考设置](https://platform.claude.com/docs/en/build-with-claude/thinking)与[effort](https://platform.claude.com/docs/en/build-with-claude/effort)：原生消息结构及模型差异。
- [Grok推理参数](https://docs.x.ai/developers/model-capabilities/text/reasoning)：版本相关的effort。
- [Gemini兼容入口](https://ai.google.dev/gemini-api/docs/openai)和[GenerateContent](https://ai.google.dev/api/generate-content)：兼容字段、原生消息与生成选项。
- [Qwen兼容接口](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)和[结构化输出](https://help.aliyun.com/zh/model-studio/qwen-structured-output)：REST思考字段与JSON适用边界。
- [4Router模型元数据](https://docs.4router.net/models.json)：候选推理等级；账户权限仍以实际请求为准。
- [DeepSeek思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)：思考开关与effort映射。

不同模型对同一强度名称的定义可能不同，论文比较应记录API协议、模型ID、实际返回ID、所选强度、预算和调用成本。
