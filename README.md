# pairwise-npc-schedule-skip

## 领域

游戏、图形与多媒体

## 主题

NPC 日程与时间跳跃补算

## 说明

分批补算、区域就绪与占用令牌。

初始状态下 `gameplay/npc_schedule_skip.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
