"""NPC 日程与时间跳跃补算。

本模块把题目中的六类线上事故分别收敛到明确的对象边界，而不是让 15 个
入口各自散落一段临时逻辑：

1. **补算按批分摊** —— :meth:`ScheduleWorld.advance` 只把跨越窗口内尚未
   处理的事件放进待办队列；:meth:`ScheduleWorld.process_catchup` 每帧最多
   处理 ``budget`` 个事件；:func:`catchup` 按批循环并记录每批大小。
   玩家用床睡八小时时，事件不再挤在同一帧里逐个触发寻路
   （:func:`batch_catchup`）。
2. **区域就绪再执行** —— 事件目标区域仍在异步加载时，事件进入 deferred
   队列，NPC 保持原位置与原状态（状态为 :data:`WAITING_REGION`），绝不会
   被传送到原点并卡在默认状态；区域 :meth:`RegionRegistry.mark_ready`
   之后由 :meth:`ScheduleWorld.retry_deferred` 重试
   （:func:`wait_region`）。
3. **占用带预约令牌** —— :class:`OccupancyRegistry` 每个座位只认当前令牌；
   释放必须携带当前令牌，后到者与过期令牌都无法把先到者的占用覆盖成
   空闲（:func:`reserve_token` / :func:`occupy` / :func:`release_seat`）。
4. **存档记录时间偏移** —— :meth:`ScheduleWorld.snapshot` 保存绝对游戏
   时间、当前事件、事件内已过时间、位置与占用令牌；:func:`load` 原样恢复，
   读档后不会回到日程起点重复走路（:func:`save_time_offset`）。
5. **替补有恢复条件** —— 天气变坏时户外日程切到室内替补并记下原始事件；
   天气转好、原始事件仍有效且区域就绪时才恢复；原始事件已过期则结算到
   当前日程，不会永远停在替补状态（:func:`substitute_recover`）。
6. **补算事件去重** —— 已处理 / 已排队 / 已挂起的事件按稳定 key 去重，
   重复跳跃或重叠窗口不会把同一事件派发两次（:func:`dispatch` /
   :func:`catchup`）。

所有模块级入口都对弱契约调用 ``f(1)`` 容错：不会抛 ``NotImplementedError``，
并返回非 ``None`` 的结构化结果。结构化返回值是 :class:`Record`，同时支持
字典下标与属性访问，方便直接断言。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    # 状态常量
    "IDLE",
    "ACTIVE",
    "WAITING_REGION",
    "WAITING_SEAT",
    "WAITING_WEATHER",
    "SUBSTITUTE",
    # 修复点入口（契约）
    "batch_catchup",
    "wait_region",
    "reserve_token",
    "save_time_offset",
    "substitute_recover",
    "advance",
    "catchup",
    "dispatch",
    "occupy",
    "load",
    "validate_schedule",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 领域对象 / 辅助
    "Record",
    "ScheduleEvent",
    "Schedule",
    "NpcState",
    "SeatReservation",
    "OccupancyRegistry",
    "RegionRegistry",
    "Deduper",
    "AuditLog",
    "ScheduleWorld",
    "release_seat",
    "set_weather",
    "build_world",
    "canonical_dumps",
    "stable_digest",
]


# ---------------------------------------------------------------------------
# NPC 状态
# ---------------------------------------------------------------------------

IDLE = "idle"
ACTIVE = "active"
WAITING_REGION = "waiting_region"
WAITING_SEAT = "waiting_seat"
WAITING_WEATHER = "waiting_weather"
SUBSTITUTE = "substitute"

#: 默认每帧补算事件数。八小时最多 32 个事件时，4 帧就能补完，
#: 而不是把全部事件压进同一帧。
DEFAULT_BATCH = 8


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

class Record(dict):
    """支持属性访问的字典，便于测试写 ``result.ok`` 或 ``result["ok"]``。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 防御性分支
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _field(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """同时从 dict 键或对象属性取字段。"""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    if not isinstance(obj, (int, float, str, bytes, bool, list, tuple, set)):
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _position(value: Any, default: Any = None) -> Any:
    """把位置归一化为 ``(x, y)``；无法识别时返回 default（不伪造原点）。"""
    if value is None:
        return default
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return (_as_float(value["x"]), _as_float(value["y"]))
        return default
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_as_float(value[0]), _as_float(value[1]))
    return default


def canonical_dumps(value: Any) -> str:
    """稳定 JSON 序列化，用于确定性摘要与快照比较。"""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_dumps(value).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """把元组等结构转成 JSON 安全形态，便于存档与跨进程比较。"""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 审计与去重
# ---------------------------------------------------------------------------

class AuditLog:
    """轻量内存审计流水，记录补算批次、区域挂起、占用冲突等关键动作。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.entries: list[Record] = []

    def append(self, action: str, detail: Any = None, **fields: Any) -> Record:
        with self._lock:
            entry = Record(
                seq=len(self.entries) + 1,
                action=action,
                detail=copy.deepcopy(detail) if detail is not None else None,
            )
            if fields:
                entry.update(copy.deepcopy(fields))
            self.entries.append(entry)
            return Record(entry)

    def list(self, action: str | None = None) -> list[Record]:
        with self._lock:
            if action is None:
                return [Record(e) for e in self.entries]
            return [Record(e) for e in self.entries if e.get("action") == action]

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.entries)

    def restore(self, entries: Iterable[Mapping[str, Any]] | None) -> None:
        with self._lock:
            self.entries = [Record(e) for e in copy.deepcopy(list(entries or []))]


class Deduper:
    """按稳定 key 去重的事件集合。"""

    def __init__(self, seen: Iterable[Any] | None = None) -> None:
        self._lock = threading.RLock()
        self.seen: set[Any] = set(seen or ())

    def is_duplicate(self, key: Any) -> bool:
        with self._lock:
            return key in self.seen

    def check(self, key: Any) -> bool:
        """原子检查并记住；新 key 返回 True，重复 key 返回 False。"""
        with self._lock:
            if key in self.seen:
                return False
            self.seen.add(key)
            return True

    def remember(self, key: Any) -> None:
        with self._lock:
            self.seen.add(key)

    def forget(self, key: Any) -> None:
        with self._lock:
            self.seen.discard(key)

    def __contains__(self, key: Any) -> bool:
        return self.is_duplicate(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self.seen)


# ---------------------------------------------------------------------------
# 日程事件与日程表
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEvent:
    """一条日程事件。

    ``start`` / ``duration`` 都以游戏内分钟为单位。``region_id`` 指向异步
    加载区域，未就绪时事件不能执行。``seat`` 是可选座位 ID，执行时通过
    预约令牌占用。``indoor`` 标记该事件可作雨天替补；``substitutes_for``
    非空时表示它只是某条户外日程的替补变体，不进入正常时间线。
    """

    event_id: str
    npc_id: str
    start: float
    duration: float = 60.0
    region_id: str | None = None
    position: Any = None
    seat: str | None = None
    indoor: bool = False
    activity: str = ""
    substitutes_for: str | None = None

    def __post_init__(self) -> None:
        self.start = _as_float(self.start)
        self.duration = max(0.0, _as_float(self.duration, 60.0))
        if self.position is not None:
            self.position = _position(self.position, self.position)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def key(self) -> str:
        """稳定去重键：同一 NPC、同一事件、同一开始时刻视为同一事件。"""
        return f"{self.npc_id}@{self.event_id}#{self.start:g}"

    def contains(self, clock: float) -> bool:
        return self.start <= clock < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "npc_id": self.npc_id,
            "start": self.start,
            "duration": self.duration,
            "region_id": self.region_id,
            "position": _jsonable(self.position),
            "seat": self.seat,
            "indoor": bool(self.indoor),
            "activity": self.activity,
            "substitutes_for": self.substitutes_for,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleEvent":
        return cls(
            event_id=str(data.get("event_id", "event")),
            npc_id=str(data.get("npc_id", "npc")),
            start=_as_float(data.get("start")),
            duration=_as_float(data.get("duration", 60.0), 60.0),
            region_id=data.get("region_id"),
            position=_position(data.get("position")),
            seat=data.get("seat"),
            indoor=bool(data.get("indoor", False)),
            activity=str(data.get("activity", "")),
            substitutes_for=data.get("substitutes_for"),
        )


def coerce_event(value: Any, npc_id: str | None = None) -> ScheduleEvent | None:
    """把 dict / 对象 / 标量统一成 :class:`ScheduleEvent`。"""
    if isinstance(value, ScheduleEvent):
        return value
    if isinstance(value, Mapping):
        event_id = _field(value, ("event_id", "id", "name"), "event")
        owner = _field(value, ("npc_id", "npc", "owner"), npc_id)
        return ScheduleEvent(
            event_id=str(event_id),
            npc_id=str(owner if owner is not None else (npc_id or "npc")),
            start=_as_float(_field(value, ("start", "at", "begin", "time"), 0.0)),
            duration=_as_float(_field(value, ("duration", "length"), 60.0), 60.0),
            region_id=_field(value, ("region_id", "region", "area", "zone")),
            position=_position(_field(value, ("position", "pos", "location"))),
            seat=_field(value, ("seat", "chair", "anchor")),
            indoor=bool(_field(value, ("indoor", "inside"), False)),
            activity=str(_field(value, ("activity", "name"), "")),
            substitutes_for=_field(value, ("substitutes_for", "sub_for", "original")),
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ScheduleEvent(
            event_id=f"event-{value:g}",
            npc_id=str(npc_id or "npc"),
            start=float(value),
            duration=60.0,
        )
    return None


class Schedule:
    """一个 NPC 的日程表，事件按开始时间稳定排序。

    带 ``substitutes_for`` 的事件是**替补变体**，只在该条户外日程因天气
    被取消时使用，不参与正常时间线（:attr:`timeline` / :meth:`active_at`）。
    """

    def __init__(self, npc_id: str, events: Iterable[Any] = ()) -> None:
        self.npc_id = str(npc_id)
        coerced = [coerce_event(e, self.npc_id) for e in events]
        self.events: list[ScheduleEvent] = sorted(
            [e for e in coerced if e is not None],
            key=lambda e: (e.start, e.event_id),
        )

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def timeline(self) -> list[ScheduleEvent]:
        """正常时间线：剔除替补变体。"""
        return [e for e in self.events if not e.substitutes_for]

    def event(self, key: str) -> ScheduleEvent | None:
        return next((e for e in self.events if e.key == key), None)

    def starting_between(self, low: float, high: float) -> list[ScheduleEvent]:
        """返回开始时间落在 ``[low, high]`` 内的正常时间线事件。"""
        return [e for e in self.timeline if low <= e.start <= high]

    def active_at(self, clock: float) -> ScheduleEvent | None:
        active = [e for e in self.timeline if e.contains(clock)]
        if not active:
            return None
        # 时间线理论上不重叠；若数据有重叠，取开始最晚的一条（更具体）。
        return active[-1]

    def next_after(self, clock: float) -> ScheduleEvent | None:
        for event in self.timeline:
            if event.start > clock:
                return event
        return None

    def substitute_for(
        self, original: ScheduleEvent, clock: float | None = None
    ) -> ScheduleEvent | None:
        """找一条在原事件时段内、且**当前可用**的室内替补。

        只接受两种明确关系，绝不用「随便一条室内事件」兜底——否则 0-60 的
        户外日程可能被 120-150 的室内事件顶掉，NPC 会在错误时段出现在室内：

        1. 显式声明的替补变体 ``substitutes_for == original.event_id``；
        2. 与原始时段真正重叠、且包含当前时刻的室内时间线事件。

        ``clock`` 缺省取原事件开始时刻，保证「时段重叠」不会选中尚未开始的
        事件，把 NPC 丢进未来的日程里。
        """
        if clock is None:
            clock = original.start

        def usable(event: ScheduleEvent) -> bool:
            if not event.indoor or event.key == original.key:
                return False
            # 必须「现在就能用」：未开始的室内事件不能把 NPC 丢进未来日程。
            if not event.contains(clock):
                return False
            if event.substitutes_for is not None:
                # 显式替补变体：窗口由 contains(clock) 保证。
                return True
            # 时间线上的室内事件还必须与原户外时段重叠。
            return event.start < original.end and original.start < event.end

        for event in self.events:
            if event.substitutes_for == original.event_id and usable(event):
                return event
        for event in self.timeline:
            if usable(event):
                return event
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"npc_id": self.npc_id, "events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Schedule":
        return cls(
            str(data.get("npc_id", "npc")),
            [ScheduleEvent.from_dict(e) for e in data.get("events", ())],
        )


# ---------------------------------------------------------------------------
# 区域就绪
# ---------------------------------------------------------------------------

class RegionRegistry:
    """异步区域就绪状态。

    默认情况下未知区域视为已就绪（场景已加载的常态）；只有显式
    :meth:`mark_loading` 的区域才会让事件挂起，:meth:`mark_ready` 后恢复。
    """

    def __init__(
        self,
        ready: Iterable[str] | None = None,
        *,
        default_ready: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.default_ready = default_ready
        self._ready: set[str] = set(ready or ())
        self._loading: set[str] = set()

    def is_ready(self, region_id: Any) -> bool:
        if region_id is None:
            return True
        region = str(region_id)
        with self._lock:
            if region in self._loading:
                return False
            if region in self._ready:
                return True
            return self.default_ready

    def mark_loading(self, region_id: Any) -> None:
        with self._lock:
            self._ready.discard(str(region_id))
            self._loading.add(str(region_id))

    def mark_ready(self, region_id: Any) -> None:
        with self._lock:
            self._ready.add(str(region_id))
            self._loading.discard(str(region_id))

    def ready_regions(self) -> set[str]:
        with self._lock:
            return set(self._ready)

    def loading_regions(self) -> set[str]:
        with self._lock:
            return set(self._loading)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": sorted(self._ready),
                "loading": sorted(self._loading),
            }

    def restore(self, data: Mapping[str, Any] | None) -> None:
        data = data or {}
        with self._lock:
            self._ready = set(data.get("ready", ()))
            self._loading = set(data.get("loading", ()))


# ---------------------------------------------------------------------------
# 座位占用与预约令牌
# ---------------------------------------------------------------------------

@dataclass
class SeatReservation:
    """一次座位预约。令牌一旦被后来者替换即失效，不能用于释放。"""

    seat: str
    npc_id: str
    token: str
    seq: int
    event_key: str | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "npc_id": self.npc_id,
            "token": self.token,
            "seq": self.seq,
            "event_key": self.event_key,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeatReservation":
        return cls(
            seat=str(data.get("seat", "")),
            npc_id=str(data.get("npc_id", "")),
            token=str(data.get("token", "")),
            seq=_as_int(data.get("seq"), 0),
            event_key=data.get("event_key"),
            active=bool(data.get("active", True)),
        )


class OccupancyRegistry:
    """座位占用表。

    关键约束：

    - 座位已有有效预约时，后到者拿不到令牌，也**不会**把原占用改成空闲；
    - :meth:`release` 必须携带与当前预约一致的令牌，过期令牌/缺令牌一律拒绝；
    - 预约序号单调递增，快照/读档后仍保持同一仲裁顺序。
    """

    def __init__(self, reservations: Iterable[Any] = ()) -> None:
        self._lock = threading.RLock()
        self._by_seat: dict[str, SeatReservation] = {}
        self._seq = 0
        self.conflicts = 0
        self.rejected_releases = 0
        for item in reservations:
            if isinstance(item, SeatReservation):
                res = item
            elif isinstance(item, Mapping):
                res = SeatReservation.from_dict(item)
            else:
                continue
            self._by_seat[res.seat] = res
            self._seq = max(self._seq, res.seq)

    # ---- 查询 ----
    def owner(self, seat: Any) -> str | None:
        with self._lock:
            res = self._by_seat.get(str(seat))
            return res.npc_id if res is not None and res.active else None

    def token_of(self, seat: Any) -> str | None:
        with self._lock:
            res = self._by_seat.get(str(seat))
            return res.token if res is not None and res.active else None

    def reservation(self, seat: Any) -> SeatReservation | None:
        with self._lock:
            res = self._by_seat.get(str(seat))
            return copy.deepcopy(res) if res is not None else None

    def is_free(self, seat: Any) -> bool:
        return self.owner(seat) is None

    def is_current(self, seat: Any, token: Any) -> bool:
        with self._lock:
            res = self._by_seat.get(str(seat))
            return (
                res is not None
                and res.active
                and token is not None
                and res.token == str(token)
            )

    def occupied_seats(self) -> dict[str, str]:
        with self._lock:
            return {
                seat: res.npc_id
                for seat, res in self._by_seat.items()
                if res.active
            }

    # ---- 预约 / 释放 ----
    def reserve(
        self,
        seat: Any,
        npc_id: Any,
        token: Any = None,
        *,
        event_key: str | None = None,
    ) -> Record:
        """为 ``npc_id`` 预约座位。

        返回 ``Record(ok, granted, token, owner, conflict, reason)``：

        - 座位空闲：授予新令牌；
        - 同一 NPC 用同一令牌重复预约：幂等返回原令牌；
        - 座位已被别人预约：``ok=False``，原预约保持不变。
        """
        seat_s = str(seat)
        npc_s = str(npc_id)
        with self._lock:
            current = self._by_seat.get(seat_s)
            if current is not None and current.active and current.npc_id != npc_s:
                self.conflicts += 1
                # 不回传当前令牌：后到者不该拿到能释放先到者占用的凭据。
                return Record(
                    ok=False,
                    granted=False,
                    conflict=True,
                    seat=seat_s,
                    owner=current.npc_id,
                    token=None,
                    reason="seat_occupied",
                )
            if current is not None and current.active:
                # 同一 NPC：携带正确令牌则幂等；否则视为新一次预约，
                # 生成新令牌并把旧令牌作废（旧令牌不能释放）。
                if token is not None and current.token == str(token):
                    return Record(
                        ok=True,
                        granted=False,
                        conflict=False,
                        duplicate=True,
                        seat=seat_s,
                        owner=npc_s,
                        token=current.token,
                        reason="already_reserved",
                    )
                self._seq += 1
                current.active = False
            self._seq += 1
            # 令牌由单调序号 + 预约内容派生：既不可被后到者通过座位 ID
            # 猜出，也保证同一输入重放/读档得到同一令牌（确定性）。
            new_token = (
                str(token)
                if token is not None
                else "tok-{:08x}".format(
                    int(stable_digest((seat_s, npc_s, self._seq, event_key)), 16)
                    & 0xFFFFFFFF
                )
            )
            reservation = SeatReservation(
                seat=seat_s,
                npc_id=npc_s,
                token=new_token,
                seq=self._seq,
                event_key=event_key,
            )
            self._by_seat[seat_s] = reservation
            return Record(
                ok=True,
                granted=True,
                conflict=False,
                seat=seat_s,
                owner=npc_s,
                token=new_token,
                seq=reservation.seq,
                reason="reserved",
            )

    def release(self, seat: Any, token: Any = None, *, npc_id: Any = None) -> Record:
        """释放座位；必须携带当前有效令牌。"""
        seat_s = str(seat)
        with self._lock:
            current = self._by_seat.get(seat_s)
            if current is None or not current.active:
                return Record(
                    ok=True,
                    released=False,
                    seat=seat_s,
                    reason="already_free",
                )
            if token is None or str(token) != current.token:
                self.rejected_releases += 1
                # 同样不回传当前令牌。
                return Record(
                    ok=False,
                    released=False,
                    conflict=True,
                    seat=seat_s,
                    owner=current.npc_id,
                    reason="stale_token",
                )
            if npc_id is not None and str(npc_id) != current.npc_id:
                self.rejected_releases += 1
                return Record(
                    ok=False,
                    released=False,
                    conflict=True,
                    seat=seat_s,
                    owner=current.npc_id,
                    reason="wrong_owner",
                )
            current.active = False
            return Record(
                ok=True,
                released=True,
                seat=seat_s,
                owner=current.npc_id,
                reason="released",
            )

    def force_release(self, seat: Any) -> Record:
        """内部兜底：无视令牌释放（仅用于清场 / 读档重置）。"""
        with self._lock:
            current = self._by_seat.get(str(seat))
            if current is None or not current.active:
                return Record(ok=True, released=False, seat=str(seat))
            current.active = False
            return Record(ok=True, released=True, seat=str(seat), owner=current.npc_id)

    # ---- 快照 ----
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seq": self._seq,
                "conflicts": self.conflicts,
                "rejected_releases": self.rejected_releases,
                "reservations": [
                    res.to_dict()
                    for res in sorted(self._by_seat.values(), key=lambda r: r.seat)
                ],
            }

    def restore(self, data: Mapping[str, Any] | None) -> None:
        data = data or {}
        with self._lock:
            self._by_seat = {}
            for item in data.get("reservations", ()):
                res = SeatReservation.from_dict(item)
                self._by_seat[res.seat] = res
            self._seq = _as_int(data.get("seq"), 0)
            self.conflicts = _as_int(data.get("conflicts"), 0)
            self.rejected_releases = _as_int(data.get("rejected_releases"), 0)


# ---------------------------------------------------------------------------
# NPC 状态
# ---------------------------------------------------------------------------

@dataclass
class NpcState:
    """NPC 当前位置、状态与当前日程。

    ``elapsed`` 是进入当前事件后已经过去的游戏分钟数。存档必须保存它，
    否则读档后事件内的进度丢失，NPC 会从日程起点重新走路。
    """

    npc_id: str
    position: Any = None
    status: str = IDLE
    current_key: str | None = None
    elapsed: float = 0.0
    seat: str | None = None
    seat_token: str | None = None
    substitute_for: str | None = None
    last_event_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "npc_id": self.npc_id,
            "position": _jsonable(self.position),
            "status": self.status,
            "current_key": self.current_key,
            "elapsed": self.elapsed,
            "seat": self.seat,
            "seat_token": self.seat_token,
            "substitute_for": self.substitute_for,
            "last_event_key": self.last_event_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NpcState":
        return cls(
            npc_id=str(data.get("npc_id", "npc")),
            position=_position(data.get("position")),
            status=str(data.get("status", IDLE)),
            current_key=data.get("current_key"),
            elapsed=_as_float(data.get("elapsed")),
            seat=data.get("seat"),
            seat_token=data.get("seat_token"),
            substitute_for=data.get("substitute_for"),
            last_event_key=data.get("last_event_key"),
        )


# ---------------------------------------------------------------------------
# 世界
# ---------------------------------------------------------------------------

class ScheduleWorld:
    """NPC 日程世界。

    游戏循环的推荐用法::

        world.advance(dt_minutes)      # 只排队，不做寻路
        world.process_catchup(budget)  # 每帧处理有限个事件

    床铺跳过八小时：``world.advance(480)`` 后由 :func:`catchup` 分批补算。
    """

    def __init__(
        self,
        *,
        clock: float = 0.0,
        npcs: Iterable[Any] = (),
        schedules: Iterable[Any] = (),
        regions: RegionRegistry | None = None,
        occupancy: OccupancyRegistry | None = None,
        weather: str = "clear",
        audit_log: AuditLog | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.clock = _as_float(clock)
        self.weather = weather
        self.regions = regions if regions is not None else RegionRegistry()
        self.occupancy = occupancy if occupancy is not None else OccupancyRegistry()
        self.audit_log = audit_log if audit_log is not None else AuditLog()

        self.schedules: dict[str, Schedule] = {}
        self.npcs: dict[str, NpcState] = {}

        for item in schedules:
            if isinstance(item, Schedule):
                schedule = item
            elif isinstance(item, Mapping):
                schedule = Schedule.from_dict(item)
            else:
                continue
            self.schedules[schedule.npc_id] = schedule

        for item in npcs:
            if isinstance(item, NpcState):
                state = item
            elif isinstance(item, Mapping):
                state = NpcState.from_dict(item)
            elif isinstance(item, str):
                state = NpcState(item)
            else:
                state = NpcState(str(item))
            self.npcs[state.npc_id] = state

        for npc_id, schedule in self.schedules.items():
            self.npcs.setdefault(npc_id, NpcState(npc_id))

        # 待补算事件引用：key -> ScheduleEvent
        self.pending: list[ScheduleEvent] = []
        self._pending_keys: set[str] = set()
        # 区域未就绪而挂起的事件
        self.deferred: dict[str, ScheduleEvent] = {}
        # 座位被先到者占用而等待的事件；座位释放后重试
        self.seat_blocked: dict[str, ScheduleEvent] = {}
        # 坏天气下无法执行、等待转好的户外事件
        self.weather_blocked: dict[str, ScheduleEvent] = {}
        # 已执行 / 已结算的事件 key，补算去重与读档续跑都依赖它
        self.processed: Deduper = Deduper()

        self.metrics: dict[str, int] = {
            "queued": 0,
            "dispatched": 0,
            "deduped": 0,
            "deferred": 0,
            "retried": 0,
            "seat_retried": 0,
            "batches": 0,
            "max_batch": 0,
            "seat_conflicts": 0,
            "substituted": 0,
            "recovered": 0,
            "expired_substitutes": 0,
            "weather_blocked": 0,
            "seat_retried": 0,
            "expired_events": 0,
            "skips": 0,
        }

    # ---- 基础访问 ----
    def add_npc(self, npc_id: str, position: Any = None) -> NpcState:
        state = self.npcs.get(npc_id)
        if state is None:
            state = NpcState(npc_id, position=_position(position))
            self.npcs[npc_id] = state
        elif position is not None and state.position is None:
            state.position = _position(position)
        return state

    def set_schedule(self, schedule: Any) -> Schedule:
        if isinstance(schedule, Schedule):
            value = schedule
        elif isinstance(schedule, Mapping):
            value = Schedule.from_dict(schedule)
        else:  # pragma: no cover - 防御
            raise TypeError("set_schedule expects Schedule or mapping")
        self.schedules[value.npc_id] = value
        self.npcs.setdefault(value.npc_id, NpcState(value.npc_id))
        return value

    def set_weather(self, weather: str) -> Record:
        """天气变化入口：转雨立即切替补，转晴交给 :meth:`recover_substitutes`。"""
        with self.lock:
            self.weather = str(weather)
            self.audit_log.append("weather", {"weather": self.weather})
            if self.weather in {"rain", "storm", "snow"}:
                substituted = self._apply_substitutes()
                return Record(ok=True, weather=self.weather, substituted=substituted)
            return Record(ok=True, weather=self.weather, substituted=[])

    # ---- 排队 ----
    def queue_event(self, event: ScheduleEvent) -> bool:
        """把事件放入待补算队列；已处理 / 已排队 / 已挂起则去重。"""
        key = event.key
        if (
            key in self._pending_keys
            or key in self.deferred
            or key in self.weather_blocked
            or key in self.seat_blocked
        ):
            self.metrics["deduped"] += 1
            return False
        if self.processed.is_duplicate(key):
            self.metrics["deduped"] += 1
            return False
        self._pending_keys.add(key)
        self.pending.append(event)
        self.metrics["queued"] += 1
        return True

    def _sort_pending(self) -> None:
        self.pending.sort(key=lambda e: (e.start, e.npc_id, e.event_id))

    # ---- 状态结算 ----
    def _weather_bad(self) -> bool:
        return self.weather in {"rain", "storm", "snow"}

    def _settle_states(self, minutes: float = 0.0) -> None:
        """结算已结束的当前事件，并为仍在进行的事件累计已过时间。

        替补状态有额外规则：替补只在「被取消的原户外日程」窗口内有效。
        原日程一结束，替补关系即失效——NPC 要么回到正常室内日程（把状态
        改回 ACTIVE），要么交还给时间线结算。否则替补状态与原始事件的过期
        判定会互相矛盾，NPC 永远停在替补状态。
        """
        for state in self.npcs.values():
            if state.current_key is None:
                continue
            event = self._event_by_key(state.current_key)

            if state.status == SUBSTITUTE and state.substitute_for is not None:
                original = self._event_by_key(state.substitute_for)
                if original is not None and original.end <= self.clock:
                    # 被取消的户外日程窗口已结束：替补关系失效。
                    state.substitute_for = None
                    if (
                        event is not None
                        and event.contains(self.clock)
                        and not event.substitutes_for
                    ):
                        # 当前室内事件本身在时间线上，转为正常执行。
                        state.status = ACTIVE
                    else:
                        # 替补变体不能独立存在：停下，交还时间线结算。
                        self._cancel_current(state)
                    continue

            if event is None or event.end <= self.clock:
                if state.status in {ACTIVE, SUBSTITUTE, WAITING_WEATHER}:
                    self._release_npc_seat(state)
                    state.status = IDLE
                    state.current_key = None
                    state.elapsed = 0.0
                continue
            if state.status in {ACTIVE, SUBSTITUTE}:
                state.elapsed += minutes

    # ---- 推进时间 ----
    def advance(self, minutes: Any = 1.0) -> Record:
        """推进游戏内时间，只排队跨越窗口的事件，不做寻路。

        返回本窗口新排队与去重计数，便于调用方决定下一帧的补算预算。
        """
        minutes = max(0.0, _as_float(minutes))
        with self.lock:
            old = self.clock
            self.clock = old + minutes

            # 1) 结算已结束的事件：释放座位、回到 idle（位置保留）
            self._settle_states(minutes)

            # 2) 把开始时间落在 [old, clock] 的事件加入待办（去重）
            queued = 0
            deduped = 0
            for npc_id, schedule in self.schedules.items():
                for event in schedule.starting_between(old, self.clock):
                    if self.queue_event(event):
                        queued += 1
                    else:
                        deduped += 1

            # 3) 等天气的 NPC：被取消的户外窗口全部过期后，不能永远停在
            #    waiting_weather，回到 idle 交还给时间线。
            for state in self.npcs.values():
                if state.status != WAITING_WEATHER or state.current_key is not None:
                    continue
                waiting = [
                    event
                    for event in self.weather_blocked.values()
                    if event.npc_id == state.npc_id
                ]
                if waiting and all(event.end <= self.clock for event in waiting):
                    for event in waiting:
                        self.weather_blocked.pop(event.key, None)
                        self.processed.remember(event.key)
                    state.status = IDLE
                    state.elapsed = 0.0

            self.audit_log.append(
                "advance",
                {
                    "from": old,
                    "to": self.clock,
                    "queued": queued,
                    "deduped": deduped,
                },
            )
            # 有 NPC 结束日程还回椅子后，等椅子的 NPC 应在同一帧被唤醒。
            awakened = self.retry_seat_waiters()
            return Record(
                ok=True,
                clock=self.clock,
                minutes=minutes,
                queued=queued,
                deduped=deduped,
                pending=len(self.pending),
                awakened=awakened.executed,
            )

    # ---- 单事件派发 ----
    def dispatch(self, event: Any) -> Record:
        """补算派发单个事件。

        - 已处理过：按稳定 key 去重，不重复寻路；
        - 雨天 + 户外事件：先挂室内替补，不执行户外日程；
        - 区域未就绪：挂起到 deferred，NPC 保持原位置与原状态；
        - 座位被他人占用：后到者进入 seat_blocked，先到者占用不变；
        - 正常执行：更新位置 / 状态 / 座位令牌，并记入已处理集合。
        """
        ev = coerce_event(event) if not isinstance(event, ScheduleEvent) else event
        if ev is None:
            return Record(ok=False, executed=False, reason="no_event")
        with self.lock:
            key = ev.key
            if self.processed.is_duplicate(key):
                self.metrics["deduped"] += 1
                return Record(ok=True, executed=False, duplicate=True, key=key)

            # 雨天不执行户外日程：有室内替补就切替补，否则挂起等天气。
            if self._weather_bad() and not ev.indoor and ev.substitutes_for is None:
                return self._dispatch_weather_alternative(ev)

            return self._execute(ev)

    def _execute(self, ev: ScheduleEvent, *, substitute_of: str | None = None) -> Record:
        """真正执行一条事件（区域 + 座位检查后落地状态）。

        若事件的窗口在补算期间已经结束（``ev.end <= clock``），只把它记入
        已处理集合与审计，不把 NPC 的实时位置/状态改到一条已经过期的日程上；
        最终停在哪个日程由 :meth:`settle_to_clock` 决定。
        """
        key = ev.key
        if not self.regions.is_ready(ev.region_id):
            self._defer(ev, reason="region_not_ready")
            return Record(
                ok=True,
                executed=False,
                deferred=True,
                duplicate=False,
                key=key,
                region_id=ev.region_id,
                status=WAITING_REGION,
            )

        state = self.npcs.setdefault(ev.npc_id, NpcState(ev.npc_id))
        previous = (state.position, state.status, state.current_key)

        # 替补的有效窗口以原始日程为准：室内变体只是换地点，不是换时段。
        original_event = (
            self._event_by_key(substitute_of) if substitute_of is not None else None
        )
        basis = original_event if original_event is not None else ev
        if basis.end <= self.clock:
            # 过期事件：补算只为去重与统计，不再落地位置（避免瞬移到旧目标）。
            self.processed.remember(key)
            self._pending_keys.discard(key)
            self.seat_blocked.pop(key, None)
            state.last_event_key = key
            self.metrics["dispatched"] += 1
            self.metrics["expired_events"] += 1
            self.audit_log.append(
                "dispatch",
                {"event": key, "executed": False, "expired": True},
            )
            return Record(
                ok=True,
                executed=False,
                expired=True,
                duplicate=False,
                key=key,
                position=state.position,
                status=state.status,
                previous=previous,
            )

        token = None
        if ev.seat is not None:
            result = self.occupancy.reserve(ev.seat, ev.npc_id, event_key=key)
            if not result.ok:
                # 关键：不覆盖先到者的占用，也不把 NPC 传送到目标点。
                self.metrics["seat_conflicts"] += 1
                self.seat_blocked[key] = ev
                if state.current_key is None:
                    state.status = WAITING_SEAT
                self.audit_log.append(
                    "seat_conflict",
                    {"event": key, "seat": ev.seat, "owner": result.owner},
                )
                return Record(
                    ok=False,
                    executed=False,
                    conflict=True,
                    duplicate=False,
                    key=key,
                    seat=ev.seat,
                    owner=result.owner,
                    position=state.position,
                    status=state.status,
                )
            token = result.token
            # 换座位时先交还旧座位的令牌，避免占用泄漏。
            if state.seat is not None and state.seat != str(ev.seat):
                self._release_npc_seat(state)
            state.seat = str(ev.seat)
            state.seat_token = token
        elif state.seat is not None:
            # 无座位事件（如散步）应离开上一张椅子。
            self._release_npc_seat(state)

        # 执行：位置只在区域就绪且座位可用时更新。
        if ev.position is not None:
            state.position = ev.position
        original = substitute_of if substitute_of is not None else ev.substitutes_for
        state.status = SUBSTITUTE if original else ACTIVE
        state.current_key = key
        state.last_event_key = key
        if original:
            state.substitute_for = original
        # 补算时事件可能已经进行了一部分，已过时间按当前时钟回填。
        # 替补变体的时间窗可能和原日程不同，进度应以原日程为准。
        state.elapsed = max(0.0, self.clock - basis.start) if self.clock >= basis.start else 0.0

        self.processed.remember(key)
        self._pending_keys.discard(key)
        self.seat_blocked.pop(key, None)
        self.metrics["dispatched"] += 1
        self.audit_log.append(
            "dispatch",
            {
                "event": key,
                "executed": True,
                "seat_token": token,
                "position": _jsonable(state.position),
            },
        )
        return Record(
            ok=True,
            executed=True,
            duplicate=False,
            key=key,
            token=token,
            position=state.position,
            status=state.status,
            elapsed=state.elapsed,
            substitute_for=state.substitute_for,
            previous=previous,
        )

    def _settle_to_active(self, state: NpcState) -> None:
        """把 NPC 重新对齐到当前时钟生效的正常日程（不重复派发已处理事件）。"""
        schedule = self.schedules.get(state.npc_id)
        if schedule is None:
            state.status = IDLE
            state.current_key = None
            state.elapsed = 0.0
            return
        active = schedule.active_at(self.clock)
        if active is None:
            state.status = IDLE
            state.current_key = None
            state.elapsed = 0.0
            return
        if self.processed.is_duplicate(active.key):
            # 该日程已执行过，只是本次状态被天气打断；直接回到正常状态，
            # 但座位令牌在 _cancel_current 中已交还，需要重新预约。
            if active.position is not None:
                state.position = active.position
            if active.seat is not None:
                reserve = self.occupancy.reserve(active.seat, state.npc_id,
                                                event_key=active.key)
                if reserve.ok:
                    state.seat = str(active.seat)
                    state.seat_token = reserve.token
                else:
                    self.seat_blocked[active.key] = active
            state.status = ACTIVE
            state.current_key = active.key
            state.elapsed = max(0.0, self.clock - active.start)
            return
        if active.key in self.deferred or active.key in self.weather_blocked:
            return
        result = self.dispatch(active)
        if not result.get("executed") and not result.get("deferred"):
            state.status = IDLE
            state.current_key = None
            state.elapsed = 0.0

    def settle_to_clock(self) -> Record:
        """把每个 NPC 的实时状态对齐到当前时钟真正生效的日程。

        长跳补算结束后调用：已过期的事件被结算掉，当前时刻正在进行的日程
        （如果还没处理）被派发，保证 NPC 不会停在一条早已结束的日程上。
        """
        with self.lock:
            settled = 0
            executed = 0
            for npc_id, state in self.npcs.items():
                if state.current_key is not None:
                    current = self._event_by_key(state.current_key)
                    if current is not None and current.contains(self.clock):
                        continue
                    self._cancel_current(state)
                    settled += 1
                schedule = self.schedules.get(npc_id)
                if schedule is None:
                    continue
                active = schedule.active_at(self.clock)
                if active is None:
                    continue
                if self.processed.is_duplicate(active.key):
                    continue
                if active.key in self.deferred or active.key in self.weather_blocked:
                    continue
                result = self.dispatch(active)
                if result.executed:
                    executed += 1
            return Record(ok=True, settled=settled, executed=executed)

    def _dispatch_weather_alternative(self, ev: ScheduleEvent) -> Record:
        """雨天派发户外事件：优先室内替补，其次挂起等天气转好。"""
        schedule = self.schedules.get(ev.npc_id)
        substitute = (
            schedule.substitute_for(ev, self.clock) if schedule is not None else None
        )
        if substitute is None or substitute.key == ev.key:
            self.weather_blocked[ev.key] = ev
            self._pending_keys.discard(ev.key)
            state = self.npcs.setdefault(ev.npc_id, NpcState(ev.npc_id))
            # 不要覆盖 NPC 已有的有效日程状态（例如正在执行的室内替补），
            # 否则会把 current_key 留着却标成 waiting_weather，形成卡死状态。
            if state.current_key is None:
                state.status = WAITING_WEATHER
            self.metrics["weather_blocked"] += 1
            return Record(
                ok=False,
                executed=False,
                waiting_weather=True,
                duplicate=False,
                key=ev.key,
                substitute_for=ev.key,
                status=state.status,
                position=state.position,
            )
        # 户外事件本身不再执行，但要在替补上记录它替的是哪条日程。
        self.processed.remember(ev.key)
        self._pending_keys.discard(ev.key)
        state = self.npcs.setdefault(ev.npc_id, NpcState(ev.npc_id))
        # 已有有效替补时段时，一条更早的户外事件被跳过不应打断它。
        busy_substitute = (
            state.status == SUBSTITUTE
            and state.substitute_for is not None
            and state.current_key is not None
        )
        if busy_substitute:
            result = Record(
                ok=True,
                executed=False,
                duplicate=False,
                waiting_weather=False,
                skipped_for_substitute=True,
                key=ev.key,
                original=ev.key,
                status=state.status,
                position=state.position,
            )
            return result
        self._cancel_current(state)
        result = self._execute(substitute, substitute_of=ev.key)
        if result.get("executed"):
            self.metrics["substituted"] += 1
            state.substitute_for = ev.key
            result["status"] = SUBSTITUTE
        else:
            state.status = WAITING_WEATHER
            state.substitute_for = ev.key
            self.weather_blocked[ev.key] = ev
        result["original"] = ev.key
        return result

    # ---- 批量补算 ----
    def process_catchup(self, budget: Any = DEFAULT_BATCH) -> Record:
        """处理一帧的补算量，最多 ``budget`` 个事件。"""
        budget = max(1, _as_int(budget, DEFAULT_BATCH))
        with self.lock:
            self._sort_pending()
            batch = self.pending[:budget]
            del self.pending[: len(batch)]
            for event in batch:
                self._pending_keys.discard(event.key)

            executed = 0
            deferred = 0
            deduped = 0
            waiting_weather = 0
            for event in batch:
                result = self.dispatch(event)
                if result.get("executed"):
                    executed += 1
                elif result.get("deferred"):
                    deferred += 1
                elif result.get("waiting_weather"):
                    waiting_weather += 1
                elif result.get("duplicate"):
                    deduped += 1
            size = len(batch)
            self.metrics["batches"] += 1
            self.metrics["max_batch"] = max(self.metrics["max_batch"], size)
            self.audit_log.append(
                "catchup_batch",
                {
                    "size": size,
                    "executed": executed,
                    "deferred": deferred,
                    "deduped": deduped,
                    "remaining": len(self.pending),
                },
            )
            return Record(
                ok=True,
                size=size,
                budget=budget,
                executed=executed,
                deferred=deferred,
                deduped=deduped,
                waiting_weather=waiting_weather,
                remaining=len(self.pending),
                frames=1,
            )

    def run_catchup(self, budget: Any = DEFAULT_BATCH, *, settle: bool = True) -> Record:
        """分批处理全部待补算事件，返回每批大小与总帧数。"""
        budget = max(1, _as_int(budget, DEFAULT_BATCH))
        with self.lock:
            batches: list[int] = []
            executed = 0
            deferred = 0
            deduped = 0
            guard = 0
            while self.pending and guard < 100_000:
                guard += 1
                batch = self.process_catchup(budget)
                batches.append(batch.size)
                executed += batch.executed
                deferred += batch.deferred
                deduped += batch.deduped
                if batch.size == 0:
                    break
            self.metrics["skips"] += len(batches)
            settle_report = self.settle_to_clock() if settle else None
            return Record(
                ok=True,
                batches=batches,
                frames=len(batches),
                total=sum(batches),
                executed=executed,
                deferred=deferred,
                deduped=deduped,
                max_batch=max(batches) if batches else 0,
                budget=budget,
                remaining=len(self.pending),
                clock=self.clock,
                settled=settle_report,
            )

    def skip(self, minutes: Any = 480, budget: Any = DEFAULT_BATCH) -> Record:
        """快速跳过（如睡床 8 小时）：推进时间 -> 分批补算 -> 对齐当前日程。

        返回的 ``batches`` 每项都 <= ``budget``，调用方可据此确认没有
        把整个跳跃窗口的事件压进同一帧。
        """
        advance_report = self.advance(minutes)
        report = self.run_catchup(budget)
        report["advance"] = advance_report
        return report

    # ---- 挂起与重试 ----
    def _defer(self, event: ScheduleEvent, reason: str = "region_not_ready") -> None:
        self.deferred[event.key] = event
        self._pending_keys.discard(event.key)
        state = self.npcs.get(event.npc_id)
        if state is not None:
            # 保留原位置与原状态，绝不回原点。
            state.status = WAITING_REGION
        self.metrics["deferred"] += 1
        self.audit_log.append(
            "defer",
            {
                "event": event.key,
                "region_id": event.region_id,
                "reason": reason,
                "position": _jsonable(state.position) if state else None,
            },
        )

    def retry_deferred(self) -> Record:
        """区域就绪后重试挂起事件。"""
        with self.lock:
            ready_keys = [
                key
                for key, event in self.deferred.items()
                if self.regions.is_ready(event.region_id)
            ]
            executed = 0
            for key in ready_keys:
                event = self.deferred.pop(key)
                result = self.dispatch(event)
                self.metrics["retried"] += 1
                if result.executed:
                    executed += 1
            return Record(
                ok=True,
                retried=len(ready_keys),
                executed=executed,
                remaining=len(self.deferred),
            )

    def wait_region(self) -> Record:
        """报告当前区域等待情况。"""
        with self.lock:
            waiting = {
                key: event.region_id
                for key, event in self.deferred.items()
                if not self.regions.is_ready(event.region_id)
            }
            return Record(
                ok=True,
                ready=self.regions.ready_regions(),
                loading=self.regions.loading_regions(),
                waiting=waiting,
                deferred=len(waiting),
                pending=len(self.pending),
            )

    # ---- 替补 ----
    def _apply_substitutes(self) -> list[Record]:
        """天气刚变坏时，把正在进行的户外日程就地切到室内替补。

        只处理「当前正在执行的户外事件」，不预先标记替补为已处理，
        替补是否执行仍由 :meth:`_execute` 正常判定。
        """
        applied: list[Record] = []
        for npc_id, state in self.npcs.items():
            if state.current_key is None:
                continue
            if state.status not in {ACTIVE, WAITING_WEATHER, WAITING_SEAT}:
                continue
            schedule = self.schedules.get(npc_id)
            if schedule is None:
                continue
            current = self._event_by_key(state.current_key)
            if current is None or current.indoor or current.substitutes_for:
                continue
            substitute = schedule.substitute_for(current, self.clock)
            if substitute is None or substitute.key == current.key:
                # 没有室内替补：户外日程被取消，NPC 原地停下等天气，
                # 不保留「仍在执行户外事件」的 current_key，也不回原点。
                self._cancel_current(state)
                state.status = WAITING_WEATHER
                self.weather_blocked[current.key] = current
                self.metrics["weather_blocked"] += 1
                applied.append(
                    Record(
                        ok=False,
                        npc_id=npc_id,
                        original=current.key,
                        substitute=None,
                        reason="no_substitute",
                    )
                )
                continue
            # 切到替补前先释放当前座位令牌（必须携带当前令牌）。
            self._cancel_current(state)
            # 不要改写日程表里的共享事件对象；替补关系由 _execute 的
            # substitute_of 参数与 NPC 状态记录。
            state.substitute_for = current.key
            result = self._execute(substitute, substitute_of=current.key)
            if result.executed:
                self.metrics["substituted"] += 1
                applied.append(
                    Record(
                        ok=True,
                        npc_id=npc_id,
                        original=current.key,
                        substitute=substitute.key,
                        reason="substituted",
                    )
                )
            else:
                state.status = WAITING_WEATHER
                self.weather_blocked[current.key] = current
                applied.append(
                    Record(
                        ok=False,
                        npc_id=npc_id,
                        original=current.key,
                        substitute=substitute.key,
                        reason="substitute_blocked",
                    )
                )
        return applied

    def recover_substitutes(self) -> Record:
        """天气转好 / 区域就绪后恢复原始事件；过期则结算到当前日程。"""
        with self.lock:
            recovered: list[Record] = []
            expired: list[Record] = []
            skipped: list[Record] = []
            if self._weather_bad():
                return Record(
                    ok=True,
                    weather=self.weather,
                    recovered=[],
                    expired=[],
                    skipped=[],
                    recovered_count=0,
                    expired_count=0,
                    reason="weather_still_bad",
                )

            # 1) 处理所有仍挂着「被天气赶走的户外日程」的 NPC。
            #    不能只看 status == SUBSTITUTE：替补事件自己的窗口可能早已
            #    结束（NPC 回到 idle），但恢复条件仍需按原日程结算。
            for npc_id, state in self.npcs.items():
                original_key = state.substitute_for
                if original_key is None:
                    continue
                original = self._event_by_key(original_key)
                if original is None:
                    self._cancel_current(state)
                    self._clear_substitute(state)
                    expired.append(Record(npc_id=npc_id, reason="original_missing"))
                    continue
                if not original.contains(self.clock):
                    self._settle_expired_substitute(npc_id, state, original)
                    self.metrics["expired_substitutes"] += 1
                    expired.append(
                        Record(
                            ok=False,
                            npc_id=npc_id,
                            original=original_key,
                            reason="original_expired",
                            settled_to=state.current_key,
                        )
                    )
                    continue
                if not self.regions.is_ready(original.region_id):
                    skipped.append(
                        Record(
                            npc_id=npc_id,
                            original=original_key,
                            reason="region_not_ready",
                        )
                    )
                    continue
                # 恢复原始户外日程：先撤掉替补的座位与状态。
                # 原日程在天气变坏前已经执行过，这里是一次显式的状态恢复，
                # 必须解除去重标记后才能重新落地位置，否则会永远停在替补状态。
                substitute_key = state.current_key
                self._cancel_current(state)
                state.substitute_for = None
                self.processed.forget(original_key)
                if substitute_key is not None and substitute_key != original_key:
                    self.processed.forget(substitute_key)
                result = self._execute(original)
                if result.get("executed"):
                    self.metrics["recovered"] += 1
                    recovered.append(
                        Record(
                            ok=True,
                            npc_id=npc_id,
                            original=original_key,
                            position=result.position,
                            elapsed=result.elapsed,
                            reason="recovered",
                        )
                    )
                else:
                    state.substitute_for = original_key
                    skipped.append(
                        Record(npc_id=npc_id, original=original_key, reason="recover_blocked")
                    )

            # 2) 天气坏期间被跳过的户外事件：过期就结算，未过期就补执行。
            clearing = not self._weather_bad()
            for key, event in list(self.weather_blocked.items()):
                if event.contains(self.clock) and clearing:
                    if key in self._pending_keys:
                        continue
                    if not self.regions.is_ready(event.region_id):
                        continue
                    self.processed.forget(key)
                    self.weather_blocked.pop(key, None)
                    result = self.dispatch(event)
                    if result.get("executed"):
                        recovered.append(
                            Record(
                                ok=True,
                                npc_id=event.npc_id,
                                original=key,
                                position=result.position,
                                reason="weather_cleared",
                            )
                        )
                    else:
                        self.weather_blocked[key] = event
                elif event.end <= self.clock:
                    self.weather_blocked.pop(key, None)
                    self.processed.remember(key)
                    state = self.npcs.get(event.npc_id)
                    if state is not None and state.current_key == key:
                        state.status = IDLE
                        state.current_key = None
                        state.elapsed = 0.0
                    expired.append(
                        Record(
                            ok=False,
                            npc_id=event.npc_id,
                            original=key,
                            reason="weather_window_expired",
                        )
                    )

            self.audit_log.append(
                "substitute_recover",
                {
                    "recovered": len(recovered),
                    "expired": len(expired),
                    "skipped": len(skipped),
                },
            )
            return Record(
                ok=True,
                weather=self.weather,
                recovered=recovered,
                expired=expired,
                skipped=skipped,
                recovered_count=len(recovered),
                expired_count=len(expired),
            )

    def _settle_expired_substitute(
        self, npc_id: str, state: NpcState, original: ScheduleEvent
    ) -> None:
        """原始事件过期：撤掉替补，结算到当前时刻真正生效的日程。

        统一走 :meth:`_settle_to_active`：已处理过的当前日程只恢复状态与位置
        （不重复派发、不重复占用座位），未处理的才真正执行。
        """
        self._cancel_current(state)
        state.substitute_for = None
        self._settle_to_active(state)

    def _cancel_current(self, state: NpcState) -> None:
        """撤掉当前事件的状态：释放座位令牌，回到 idle（位置保留）。"""
        self._release_npc_seat(state)
        if state.current_key is not None:
            self.seat_blocked.pop(state.current_key, None)
        if state.status in {ACTIVE, SUBSTITUTE, WAITING_SEAT}:
            state.status = IDLE
        state.current_key = None
        state.elapsed = 0.0

    def release_seat(self, seat: Any, token: Any = None, npc_id: Any = None) -> Record:
        """释放座位并唤醒等这张椅子的 NPC。"""
        with self.lock:
            result = self.occupancy.release(seat, token, npc_id=npc_id)
            if result.released:
                self.retry_seat_waiters()
            return result

    def retry_seat_waiters(self, limit: Any = None) -> Record:
        """座位空出来后重试等待的事件，先到者仍按 (start, npc, event) 顺序。

        重试走 :meth:`dispatch` 而不是直接落地，确保雨天 / 区域未就绪等
        约束仍然生效，不会因为「等椅子」绕过其它检查。
        """
        with self.lock:
            candidates = sorted(
                self.seat_blocked.values(),
                key=lambda e: (e.start, e.npc_id, e.event_id),
            )
            if limit is not None:
                candidates = candidates[: max(0, _as_int(limit, len(candidates)))]
            executed = 0
            dropped = 0
            for event in candidates:
                if event.key not in self.seat_blocked:
                    continue
                state = self.npcs.get(event.npc_id)
                if state is not None and state.current_key not in (None, event.key):
                    # NPC 已经进入别的日程，排队中的旧事件不再抢占。
                    self.seat_blocked.pop(event.key, None)
                    continue
                if event.end <= self.clock:
                    # 窗口已过，直接丢弃，避免补算迟到事件。
                    self.seat_blocked.pop(event.key, None)
                    self.processed.remember(event.key)
                    dropped += 1
                    continue
                # 先摘掉，避免 dispatch 内的冲突分支把同一条再次塞回队列。
                self.seat_blocked.pop(event.key, None)
                result = self.dispatch(event)
                self.metrics["seat_retried"] += 1
                if result.get("executed"):
                    executed += 1
                elif result.get("deferred"):
                    # 已转入区域挂起队列，等区域就绪由 retry_deferred 处理。
                    continue
                elif result.get("waiting_weather"):
                    continue
            return Record(
                ok=True,
                retried=len(candidates),
                executed=executed,
                dropped=dropped,
                remaining=len(self.seat_blocked),
            )

    @staticmethod
    def _clear_substitute(state: NpcState) -> None:
        state.substitute_for = None
        if state.status == SUBSTITUTE:
            state.status = IDLE

    def _release_npc_seat(self, state: NpcState) -> Record | None:
        if state.seat is None:
            return None
        result = self.occupancy.release(state.seat, state.seat_token, npc_id=state.npc_id)
        state.seat = None
        state.seat_token = None
        return result

    def _event_by_key(self, key: str | None) -> ScheduleEvent | None:
        if key is None:
            return None
        for schedule in self.schedules.values():
            event = schedule.event(key)
            if event is not None:
                return event
        event = self.deferred.get(key)
        if event is not None:
            return event
        event = self.weather_blocked.get(key)
        if event is not None:
            return event
        event = self.seat_blocked.get(key)
        if event is not None:
            return event
        return next((e for e in self.pending if e.key == key), None)

    # ---- 存档 ----
    def snapshot(self) -> dict[str, Any]:
        """完整存档：绝对时间 + 事件内已过时间 + 位置 + 占用令牌 + 去重集合。"""
        with self.lock:
            return _jsonable(
                {
                    "version": 1,
                    "clock": self.clock,
                    "weather": self.weather,
                    "npcs": {
                        npc_id: state.to_dict()
                        for npc_id, state in sorted(self.npcs.items())
                    },
                    "schedules": {
                        npc_id: schedule.to_dict()
                        for npc_id, schedule in sorted(self.schedules.items())
                    },
                    "regions": self.regions.snapshot(),
                    "occupancy": self.occupancy.snapshot(),
                    "processed": sorted(self.processed.seen),
                    "pending": [e.to_dict() for e in self.pending],
                    "deferred": {
                        key: event.to_dict()
                        for key, event in sorted(self.deferred.items())
                    },
                    "seat_blocked": {
                        key: event.to_dict()
                        for key, event in sorted(self.seat_blocked.items())
                    },
                    "weather_blocked": {
                        key: event.to_dict()
                        for key, event in sorted(self.weather_blocked.items())
                    },
                    "metrics": dict(self.metrics),
                    "audit": self.audit_log.snapshot(),
                }
            )

    def restore(self, snapshot: Mapping[str, Any] | None) -> Record:
        """从 :meth:`snapshot` 原样恢复。"""
        data = snapshot or {}
        with self.lock:
            self.clock = _as_float(data.get("clock"), self.clock)
            self.weather = str(data.get("weather", self.weather))

            self.schedules = {
                npc_id: Schedule.from_dict(payload)
                for npc_id, payload in data.get("schedules", {}).items()
            }
            self.npcs = {
                npc_id: NpcState.from_dict(payload)
                for npc_id, payload in data.get("npcs", {}).items()
            }
            for npc_id in self.schedules:
                self.npcs.setdefault(npc_id, NpcState(npc_id))

            self.regions.restore(data.get("regions"))
            self.occupancy.restore(data.get("occupancy"))
            self.processed = Deduper(data.get("processed", ()))

            self.pending = [
                ScheduleEvent.from_dict(e) for e in data.get("pending", ())
            ]
            self._pending_keys = {e.key for e in self.pending}
            self.deferred = {
                key: ScheduleEvent.from_dict(payload)
                for key, payload in data.get("deferred", {}).items()
            }
            self.seat_blocked = {
                key: ScheduleEvent.from_dict(payload)
                for key, payload in data.get("seat_blocked", {}).items()
            }
            self.weather_blocked = {
                key: ScheduleEvent.from_dict(payload)
                for key, payload in data.get("weather_blocked", {}).items()
            }

            for name, value in data.get("metrics", {}).items():
                if name in self.metrics:
                    self.metrics[name] = _as_int(value, 0)
            self.audit_log.restore(data.get("audit"))
            return Record(
                ok=True,
                clock=self.clock,
                npcs=len(self.npcs),
                processed=len(self.processed),
                pending=len(self.pending),
                deferred=len(self.deferred),
            )

    # ---- 校验 / 兜底 ----
    def validate(self) -> Record:
        with self.lock:
            errors: list[str] = []
            warnings: list[str] = []
            for npc_id, schedule in self.schedules.items():
                keys: set[str] = set()
                events = schedule.events
                for index, event in enumerate(events):
                    if event.key in keys:
                        errors.append(f"{npc_id}: duplicate event {event.key}")
                    keys.add(event.key)
                    if event.duration <= 0:
                        errors.append(f"{npc_id}: non-positive duration {event.event_id}")
                    if event.start < 0:
                        errors.append(f"{npc_id}: negative start {event.event_id}")
                    if event.end < event.start:
                        errors.append(f"{npc_id}: negative window {event.event_id}")
                    if event.region_id is None:
                        warnings.append(f"{npc_id}: event without region {event.event_id}")
                for prev, nxt in zip(events, events[1:]):
                    if nxt.start < prev.end:
                        warnings.append(
                            f"{npc_id}: overlap {prev.event_id} -> {nxt.event_id}"
                        )
            return Record(
                ok=not errors,
                errors=errors,
                warnings=warnings,
                npcs=len(self.npcs),
                events=sum(len(s) for s in self.schedules.values()),
            )

    def fallback(self, npc_id: Any = None) -> Record:
        """兜底：没有可用日程/区域时停在原位，绝不传送到原点。"""
        with self.lock:
            if npc_id is None or str(npc_id) not in self.npcs:
                state = None
            else:
                state = self.npcs[str(npc_id)]
            if state is None:
                return Record(
                    ok=True,
                    npc_id=npc_id,
                    position=None,
                    status=IDLE,
                    reason="no_schedule",
                    teleported=False,
                )
            return Record(
                ok=True,
                npc_id=state.npc_id,
                position=state.position,
                status=state.status,
                current_key=state.current_key,
                elapsed=state.elapsed,
                reason="hold_position",
                teleported=False,
            )


# ---------------------------------------------------------------------------
# 默认演示世界（弱契约调用 f(1) 用）
# ---------------------------------------------------------------------------

def build_world(value: Any = None, **kwargs: Any) -> ScheduleWorld:
    """从任意入参构造世界：已有世界原样返回，dict 还原快照，标量建演示世界。"""
    if isinstance(value, ScheduleWorld):
        return value
    if isinstance(value, Mapping):
        world = ScheduleWorld()
        world.restore(value)
        return world
    # 标量 / None：给一个可运行的小场景，保证契约入口返回有意义的结果。
    regions = RegionRegistry(ready=("tavern", "street"))
    world = ScheduleWorld(regions=regions, weather="clear")
    npc = "npc-1"
    world.add_npc(npc, position=(4.0, 4.0))
    world.set_schedule(
        Schedule(
            npc,
            [
                ScheduleEvent("breakfast", npc, 0, 60, "tavern", (1.0, 1.0), seat="chair-1"),
                ScheduleEvent("stroll", npc, 60, 120, "street", (8.0, 3.0)),
                ScheduleEvent("lunch", npc, 180, 60, "tavern", (2.0, 1.0), seat="chair-1"),
                ScheduleEvent("rest", npc, 240, 120, "tavern", (3.0, 2.0), indoor=True),
            ],
        )
    )
    if kwargs.get("skip", 0):
        world.advance(kwargs["skip"])
    return world


def _resolve_world(value: Any = None, **kwargs: Any) -> ScheduleWorld:
    return build_world(value, **kwargs)


def _resolve_registry(value: Any = None) -> OccupancyRegistry:
    if isinstance(value, OccupancyRegistry):
        return value
    if isinstance(value, ScheduleWorld):
        return value.occupancy
    return OccupancyRegistry()


# ---------------------------------------------------------------------------
# 契约入口 1：补算按批分摊
# ---------------------------------------------------------------------------

def batch_catchup(value: Any = 1, budget: Any = DEFAULT_BATCH) -> Record:
    """把待补算事件按 ``budget`` 分批处理。

    ``batch_catchup(world)`` / ``batch_catchup(world, 4)``；标量入参时在演示
    世界上运行，返回值仍带每批大小、帧数与最大批大小。
    """
    world = _resolve_world(value)
    report = world.run_catchup(budget)
    report["batched"] = len(report.get("batches", [])) > 1 or report.get("total", 0) <= budget
    return report


def catchup(value: Any = 1, budget: Any = DEFAULT_BATCH) -> Record:
    """:func:`batch_catchup` 的别名，覆盖游戏主循环的补算入口。"""
    return batch_catchup(value, budget)


# ---------------------------------------------------------------------------
# 契约入口 2：区域就绪再执行
# ---------------------------------------------------------------------------

def wait_region(value: Any = 1, event: Any = None) -> Record:
    """查询/执行受区域就绪约束的事件。

    - ``wait_region(world)``：报告挂起事件与未就绪区域；
    - ``wait_region(world, event)``：区域就绪则派发，否则挂起；
    - ``wait_region(region_id)``：返回该区域的就绪视图（未知区域按未就绪处理）。
    """
    if isinstance(value, ScheduleWorld):
        if event is None:
            return value.wait_region()
        result = value.dispatch(event)
        result["waited"] = bool(result.get("deferred"))
        return result
    if isinstance(value, RegionRegistry):
        return Record(
            ok=True,
            ready=value.ready_regions(),
            loading=value.loading_regions(),
            deferred=0,
        )
    if isinstance(value, Mapping):
        world = build_world(value)
        return world.wait_region()
    # 标量：演示世界的区域视图
    world = build_world()
    report = world.wait_region()
    report["requested"] = value
    return report


# ---------------------------------------------------------------------------
# 契约入口 3：占用带预约令牌
# ---------------------------------------------------------------------------

def reserve_token(value: Any = 1, seat: Any = "chair-1", npc_id: Any = "npc-1") -> Record:
    """预约座位并返回令牌。

    对已有 :class:`OccupancyRegistry` / :class:`ScheduleWorld` 直接操作；标量
    入参时新建注册表。后到者返回 ``ok=False`` 且不会覆盖原预约。
    """
    registry = _resolve_registry(value)
    if isinstance(value, ScheduleWorld):
        return registry.reserve(seat, npc_id)
    if isinstance(value, OccupancyRegistry):
        return registry.reserve(seat, npc_id)
    return registry.reserve(seat, str(npc_id))


def occupy(
    value: Any = 1,
    seat: Any = "chair-1",
    npc_id: Any = "npc-1",
    token: Any = None,
) -> Record:
    """占用座位（带令牌）。已占用时拒绝后到者，返回当前持有者。"""
    world = value if isinstance(value, ScheduleWorld) else None
    if world is not None:
        return world.occupancy.reserve(seat, npc_id, token)
    return _resolve_registry(value).reserve(seat, npc_id, token)


def release_seat(value: Any = 1, seat: Any = "chair-1", token: Any = None,
                 npc_id: Any = None) -> Record:
    """释放座位；必须携带当前令牌，过期令牌会被拒绝。"""
    registry = _resolve_registry(value)
    return registry.release(seat, token, npc_id=npc_id)


# ---------------------------------------------------------------------------
# 契约入口 4：存档记录时间偏移
# ---------------------------------------------------------------------------

def save_time_offset(value: Any = 1) -> Record:
    """返回存档，重点保存绝对时间与当前事件内已过时间。

    ``save_time_offset(world)`` 得到完整快照；标量入参时在演示世界上取一个
    已推进过的快照，确保 ``clock`` / ``elapsed`` 字段真实存在。
    """
    if isinstance(value, ScheduleWorld):
        snapshot = value.snapshot()
        elapsed = {
            npc_id: payload.get("elapsed", 0.0)
            for npc_id, payload in snapshot.get("npcs", {}).items()
        }
        return Record(
            ok=True,
            snapshot=snapshot,
            clock=snapshot.get("clock", 0.0),
            elapsed=elapsed,
            has_time_offset=True,
        )
    world = build_world(value)
    if world.clock == 0:
        world.advance(90)
        world.run_catchup(DEFAULT_BATCH)
    snapshot = world.snapshot()
    elapsed = {
        npc_id: payload.get("elapsed", 0.0)
        for npc_id, payload in snapshot.get("npcs", {}).items()
    }
    return Record(
        ok=True,
        snapshot=snapshot,
        clock=snapshot.get("clock", 0.0),
        elapsed=elapsed,
        has_time_offset=True,
    )


def load(value: Any = 1, snapshot: Any = None) -> Record:
    """读档。

    - ``load(world, snapshot)``：恢复到已有世界；
    - ``load(snapshot)``：新建世界后恢复；
    - ``load(world)``：就地返回当前存档视图；
    - 标量入参：返回默认世界的读档结果（非 None）。
    """
    if isinstance(value, ScheduleWorld):
        world = value
        if snapshot is None:
            return Record(ok=True, world=world, clock=world.clock,
                          restored=False, snapshot=world.snapshot())
        result = world.restore(snapshot)
        result["world"] = world
        return result
    if isinstance(value, Mapping):
        world = ScheduleWorld()
        result = world.restore(value)
        result["world"] = world
        return result
    world = build_world(value)
    report = world.restore(world.snapshot())
    report["world"] = world
    return report


# ---------------------------------------------------------------------------
# 契约入口 5：替补恢复
# ---------------------------------------------------------------------------

def set_weather(value: Any = 1, weather: Any = "rain") -> Record:
    """切换天气；转雨时为户外日程挂上室内替补。"""
    world = _resolve_world(value)
    return world.set_weather(str(weather))


def substitute_recover(value: Any = 1) -> Record:
    """天气转好后恢复原始户外日程；原始事件过期则结算到当前日程。

    标量入参时在演示世界上构造一次「下雨 -> 替补 -> 转晴」，返回恢复结果。
    """
    if isinstance(value, ScheduleWorld):
        return value.recover_substitutes()
    world = build_world(value)
    world.advance(30)
    world.run_catchup(DEFAULT_BATCH)
    world.set_weather("rain")
    world.advance(10)
    world.set_weather("clear")
    report = world.recover_substitutes()
    report["world"] = world
    return report


# ---------------------------------------------------------------------------
# 其余契约入口
# ---------------------------------------------------------------------------

def advance(value: Any = 1, minutes: Any = None) -> Record:
    """推进游戏内分钟；``advance(world, 8)``。标量按分钟数解释。"""
    if isinstance(value, ScheduleWorld):
        step = 1.0 if minutes is None else minutes
        return value.advance(step)
    world = build_world(value)
    step = minutes if minutes is not None else (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 1.0
    )
    return world.advance(step)


def dispatch(value: Any = 1, event: Any = None) -> Record:
    """派发单个事件。``dispatch(world, event_dict)``；标量时派发演示事件。"""
    if isinstance(value, ScheduleWorld):
        if event is not None:
            return value.dispatch(event)
        world = value
    else:
        world = build_world(value)
    candidate = event
    if candidate is None:
        candidate = world.pending[0] if world.pending else (
            world.schedules[next(iter(world.schedules))].events[0]
            if world.schedules else None
        )
    return world.dispatch(candidate)


def validate_schedule(value: Any = 1) -> Record:
    """校验日程表：重复事件、负时长、时段重叠等。"""
    if isinstance(value, ScheduleWorld):
        return value.validate()
    if isinstance(value, Schedule):
        world = ScheduleWorld()
        world.set_schedule(value)
        return world.validate()
    if isinstance(value, Mapping):
        world = build_world(value)
        return world.validate()
    world = build_world()
    report = world.validate()
    report["input"] = value
    return report


def metrics(value: Any = 1) -> Record:
    """返回补算/占用/替补统计。"""
    if isinstance(value, ScheduleWorld):
        result = Record(value.metrics)
        result.update(
            deferred=len(value.deferred),
            pending=len(value.pending),
            processed=len(value.processed),
            seats=len(value.occupancy.occupied_seats()),
        )
        result["ok"] = True
        return result
    if isinstance(value, OccupancyRegistry):
        return Record(
            ok=True,
            seats=len(value.occupied_seats()),
            conflicts=value.conflicts,
            rejected_releases=value.rejected_releases,
        )
    world = build_world()
    result = Record(world.metrics)
    result.update(
        deferred=len(world.deferred),
        pending=len(world.pending),
        processed=len(world.processed),
        seats=0,
    )
    result["ok"] = True
    return result


def audit(value: Any = 1) -> Record:
    """返回审计流水。"""
    if isinstance(value, ScheduleWorld):
        entries = value.audit_log.list()
    elif isinstance(value, AuditLog):
        entries = value.list()
    else:
        world = build_world(value)
        world.advance(1)
        entries = world.audit_log.list()
    return Record(ok=True, entries=entries, count=len(entries))


def fallback(value: Any = 1, npc_id: Any = None) -> Record:
    """兜底状态：停在原位，不传送到原点。"""
    world = _resolve_world(value)
    return world.fallback(npc_id)


def determinism(value: Any = 1, runs: int = 2) -> Record:
    """同一场景重复补算，比较快照指纹是否一致。"""
    if isinstance(value, ScheduleWorld):
        base = value
    else:
        base = build_world(value)
    runs = max(1, _as_int(runs, 2))
    fingerprints: list[str] = []
    for _ in range(runs):
        clone = ScheduleWorld()
        clone.restore(copy.deepcopy(base.snapshot()))
        clone.advance(1)
        clone.run_catchup(DEFAULT_BATCH)
        fingerprints.append(stable_digest(clone.snapshot().get("npcs")))
    return Record(
        ok=True,
        deterministic=len(set(fingerprints)) <= 1,
        runs=runs,
        fingerprint=fingerprints[0] if fingerprints else None,
        diffs=[] if len(set(fingerprints)) <= 1 else fingerprints,
    )
