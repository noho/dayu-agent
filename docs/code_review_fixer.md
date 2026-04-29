## 低修复风险
你充当 code review findings的fixer 角色，接下来fix code-review-0428-1727-low.md 里的findings。
medium里全部都是评估为低修复风险的finding。为了避免一次性修复太多findings 超出了你能处理的上下文范围，你先过一遍low里的条目，把所有的findings分成N批，分成多少批，以你有把握稳定修复为准，分批结果写回到low文件头部。
修复范式是先判断是不是问题（不限于bug），先给出问题诊断和修复方案，等我确认；确认后先把诊断写回low再开始修复，修复一批后输出修复报告并停下来等review，根据review意见继续修复（若还有问题）或把修复方案和修复结果写回low（若无问题）；然后修复下一批；如果实际修复中发现修复风险大于“低”，停止修复，并把finding写入medium.md。


## 中修复风险
你充当 code review findings的fixer 角色，接下来fix code-review-0428-1727-medium.md 里的findings。
medium里全部都是评估为中修复风险的finding。为了避免一次性修复太多findings 超出了你能处理的上下文范围，你先过一遍medium里的条目，把所有的findings分成N批，分成多少批，以你有把握稳定修复为准，分批结果写回到medium文件头部。
修复范式是先判断是不是问题（不限于bug），先给出问题诊断和修复方案，等我确认；确认后先把诊断写回medium再开始修复，修复一批后输出修复报告并停下来等review，根据review意见继续修复（若还有问题）或把修复方案和修复结果写回medium（若无问题）；然后修复下一批；如果实际修复中发现修复风险大于“中”，停止修复，并把finding写入high.md。


## 高修复风险
你充当 code review findings的fixer 角色，接下来fix code-review-0428-1727-high.md 里的findings。
high里全部都是评估为高修复风险的finding，修复时一定要深度思考，不要引入新bug。
修复范式是先判断是不是问题（不限于bug），先给出问题诊断和修复方案，等我确认；确认后先把诊断写回high再开始修复，修复一批后输出修复报告并停下来等review，根据review意见继续修复（若还有问题）或把修复方案和修复结果写回high（若无问题）；然后修复下一批；如果实际修复中发现修复风险过大，进入plan模式，经跟我讨论后再落地实施。


## review uncommited code
你充当 code reviewer 角色，请帮我 review 当前代码改动。
你将收到另外一个Agent对 findings 的修复报告，你要review的是另外一个Agent对findings的修复。
代码在 uncommited code 里，finding 可以根据编号在 code-review-0428-1727-high.md 中找到。
Review 要求：
- 采用 code review 模式，优先找真正会影响正确性、稳定性、可维护性的缺陷
- 对修复报告中对findings逐条review
- 重点关注：逻辑错误、边界条件、回归风险、类型问题、异常处理、测试遗漏、与现有架构约束不一致的地方
输出：
- review结束后给出review意见
- 对修复报告中的findings逐条给出review结论
- 全部findings review完毕后给出是否可commit的结论
等我给你另外一个Agent对 findings 的修复报告再开始工作。

