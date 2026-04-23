## 中修复风险
你充当 code review findings的fixer 角色，接下来fix code-review-0423-0936-medium.md 里的findings。
medium里全部都是评估为中修复风险的finding。为了避免一次性修复太多findings 超出了你能处理的上下文范围，你先过一遍medium里的条目，把所有的findings分成N批，分成多少批，以你有把握稳定修复为准，分批结果写回到medium.md 文件头部。
修复范式是先判断是不是和bug，是bug才修复；修复一批停下来等review，如果有review意见我会把意见贴给你继续修复，如果没有review意见就把修复结果写回medium，然后修复下一批；如果实际修复中发现修复风险大于“中”，停止修复，并把finding写入high.md。


## 高修复风险
你充当 code review findings的fixer 角色，接下来fix code-review-0423-0936-high.md 里的findings。
high里全部都是评估为高修复风险的finding，修复时一定要谨慎，不要引入新bug。
修复范式是先判断是不是和bug，是bug才修复；修复一个停下来等review，如果有review意见我会把意见贴给你继续修复，如果没有review意见就把修复结果写回medium，然后修复下一个；如果实际修复中发现修复风险过大，进入plan模式，经跟我讨论后再落地实施。


## review uncommited code
你充当 code reviewer 角色，review的是另外一个Agent对findings的修复。
代码在 uncommited code 里，finding 可以根据编号在 code-review-0423-0936-high.md 中找到。
等我给你另外一个Agent对 findings 的修复报告再开始工作。

