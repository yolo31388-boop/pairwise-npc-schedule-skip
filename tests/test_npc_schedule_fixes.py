"""六个修复点的行为测试（契约测试之外的正确性验证）。

覆盖：
1. batch_catchup / skip —— 八小时补算按批分摊，每批不超过预算
2. wait_region / retry_deferred —— 区域未就绪不传送，就绪后才执行
3. reserve_token / occupy / release_seat —— 座位占用带预约令牌，不互相覆盖
4. save_time_offset / load —— 存档记录绝对时间与事件内已过时间
5. substitute_recover —— 天气转好恢复原始日程，过期则结算到当前日程
6. dispatch / catchup —— 补算事件去重，重复跳跃不重复执行
"""

import copy
import json
import threading

from gameplay.npc_schedule_skip import (
    ACTIVE,
    SUBSTITUTE,
    WAITING_REGION,
    WAITING_SEAT,
    WAITING_WEATHER,
    OccupancyRegistry,
    RegionRegistry,
    Schedule,
    ScheduleEvent,
    ScheduleWorld,
    advance,
    batch_catchup,
    catchup,
    determinism,
    dispatch,
    fallback,
    load,
    metrics,
    occupy,
    release_seat,
    reserve_token,
    save_time_offset,
    substitute_recover,
    validate_schedule,
    wait_region,
)


# ---------------------------------------------------------------------------
# 1. 补算按批分摊
# ---------------------------------------------------------------------------

def _day_of_events(npc="npc", count=32, step=15, duration=15):
    return Schedule(
        npc,
        [
            ScheduleEvent(f"e{i}", npc, i * step, duration, "tavern", (float(i), 1.0))
            for i in range(count)
        ],
    )


def test_eight_hour_skip_is_split_into_bounded_batches():
    world = ScheduleWorld()
    world.set_schedule(_day_of_events(count=32))
    report = world.skip(480, budget=8)

    assert report["total"] == 32
    assert all(size <= 8 for size in report["batches"])
    assert report["frames"] == 4
    assert report["max_batch"] == 8
    assert world.pending == []


def test_single_process_catchup_respects_budget():
    world = ScheduleWorld()
    world.set_schedule(_day_of_events(count=20))
    world.advance(480)

    first = world.process_catchup(5)
    assert first["size"] == 5
    assert len(world.pending) == 15

    second = world.process_catchup(5)
    assert second["size"] == 5
    assert len(world.pending) == 10


def test_advance_only_queues_and_does_not_pathfind():
    world = ScheduleWorld()
    world.set_schedule(_day_of_events(count=32))
    report = world.advance(480)
    assert report["queued"] == 32
    assert world.metrics["dispatched"] == 0
    assert len(world.pending) == 32


def test_batch_catchup_entry_uses_world_and_reports_batches():
    world = ScheduleWorld()
    world.set_schedule(_day_of_events(count=10))
    world.advance(200)
    report = batch_catchup(world, 3)
    assert report is not None
    assert all(size <= 3 for size in report["batches"])
    assert report["frames"] == 4


# ---------------------------------------------------------------------------
# 2. 区域就绪再执行
# ---------------------------------------------------------------------------

def test_region_not_ready_keeps_npc_in_place_not_origin():
    world = ScheduleWorld()
    world.add_npc("n", position=(5.0, 6.0))
    world.set_schedule(
        Schedule("n", [ScheduleEvent("walk", "n", 0, 60, "meadow", (9.0, 9.0))])
    )
    world.regions.mark_loading("meadow")

    world.advance(1)
    report = world.run_catchup()

    state = world.npcs["n"]
    assert report["deferred"] == 1
    assert state.status == WAITING_REGION
    assert state.position == (5.0, 6.0)          # 不是 (0, 0)
    assert state.current_key is None
    assert len(world.deferred) == 1


def test_region_ready_then_retry_executes_deferred_event():
    world = ScheduleWorld()
    world.add_npc("n", position=(5.0, 6.0))
    world.set_schedule(
        Schedule("n", [ScheduleEvent("walk", "n", 0, 120, "meadow", (9.0, 9.0))])
    )
    world.regions.mark_loading("meadow")
    world.advance(1)
    world.run_catchup()
    assert world.npcs["n"].status == WAITING_REGION

    view = wait_region(world)
    assert view["deferred"] == 1
    assert "meadow" in view["loading"]

    # 区域就绪前重试：仍然挂起，位置不变
    world.retry_deferred()
    assert world.npcs["n"].status == WAITING_REGION
    assert world.npcs["n"].position == (5.0, 6.0)

    world.regions.mark_ready("meadow")
    report = world.retry_deferred()
    assert report["executed"] == 1
    assert world.npcs["n"].status == ACTIVE
    assert world.npcs["n"].position == (9.0, 9.0)


def test_unknown_region_is_ready_by_default():
    regions = RegionRegistry()
    assert regions.is_ready("never_declared") is True
    regions.mark_loading("never_declared")
    assert regions.is_ready("never_declared") is False
    regions.mark_ready("never_declared")
    assert regions.is_ready("never_declared") is True


# ---------------------------------------------------------------------------
# 3. 占用带预约令牌
# ---------------------------------------------------------------------------

def test_late_arrival_cannot_overwrite_earlier_occupant():
    occ = OccupancyRegistry()
    first = occ.reserve("chair-1", "alice")
    second = occ.reserve("chair-1", "bob")

    assert first.ok and first.granted
    assert second.ok is False and second.conflict is True
    assert second.token is None
    assert occ.owner("chair-1") == "alice"


def test_release_requires_current_token():
    occ = OccupancyRegistry()
    first = occ.reserve("chair-1", "alice")
    occ.reserve("chair-1", "bob")

    stale = occ.release("chair-1", "not-the-token")
    assert stale.released is False
    assert stale.reason == "stale_token"
    assert occ.owner("chair-1") == "alice"

    released = occ.release("chair-1", first.token)
    assert released.released is True
    assert occ.owner("chair-1") is None


def test_late_arrival_cannot_mark_earlier_occupant_free():
    """核心回归：后到者失败后，先到者的占用仍必须是「被占用」。"""
    occ = OccupancyRegistry()
    first = occ.reserve("chair-1", "alice")
    occ.reserve("chair-1", "bob")
    # 后到者无论拿什么令牌都释放不掉先到者
    assert occ.release("chair-1", None).ok is False
    assert occ.release("chair-1", "bob").ok is False
    assert occ.is_free("chair-1") is False
    assert occ.token_of("chair-1") == first.token


def test_world_seat_conflict_keeps_first_npc_seated():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("alice", [ScheduleEvent("sit", "alice", 0, 120, "tavern", (2.0, 2.0), seat="chair")])
    )
    world.set_schedule(
        Schedule("bob", [ScheduleEvent("sit", "bob", 0, 120, "tavern", (2.0, 3.0), seat="chair")])
    )
    world.advance(1)
    world.run_catchup()

    alice, bob = world.npcs["alice"], world.npcs["bob"]
    assert alice.status == ACTIVE
    assert alice.position == (2.0, 2.0)
    assert bob.status == WAITING_SEAT
    assert bob.position is None                 # 后到者没有被传送过去
    assert world.occupancy.owner("chair") == "alice"


def test_seat_handoff_when_first_npc_leaves():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("alice", [ScheduleEvent("sit", "alice", 0, 60, "tavern", (2.0, 2.0), seat="chair")])
    )
    world.set_schedule(
        Schedule("bob", [ScheduleEvent("sit", "bob", 0, 120, "tavern", (2.0, 3.0), seat="chair")])
    )
    world.advance(1)
    world.run_catchup()
    assert world.npcs["bob"].status == WAITING_SEAT

    world.advance(60)  # alice 的时段结束并交还椅子
    bob = world.npcs["bob"]
    assert bob.status == ACTIVE
    assert bob.position == (2.0, 3.0)
    assert world.occupancy.owner("chair") == "bob"


def test_reserve_token_and_release_seat_entries():
    occ = OccupancyRegistry()
    granted = reserve_token(occ, "chair-9", "npc-a")
    assert granted.ok and granted.token
    blocked = occupy(occ, "chair-9", "npc-b")
    assert blocked.ok is False and occ.owner("chair-9") == "npc-a"
    assert release_seat(occ, "chair-9", "bogus").ok is False
    assert release_seat(occ, "chair-9", granted.token).released is True
    assert occ.owner("chair-9") is None


# ---------------------------------------------------------------------------
# 4. 存档记录时间偏移
# ---------------------------------------------------------------------------

def test_snapshot_stores_elapsed_time_within_event():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("n", [ScheduleEvent("a", "n", 0, 120, "tavern", (1.0, 1.0))])
    )
    world.advance(30)
    world.run_catchup()

    saved = save_time_offset(world)
    assert saved.has_time_offset is True
    assert saved.clock == 30.0
    assert saved.elapsed["n"] == 30.0


def test_load_resumes_mid_event_instead_of_restarting():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("n", [ScheduleEvent("a", "n", 0, 120, "tavern", (1.0, 1.0))])
    )
    world.advance(30)
    world.run_catchup()
    snapshot = copy.deepcopy(world.snapshot())

    restored = ScheduleWorld()
    report = load(restored, snapshot)

    assert report.ok is True
    state = restored.npcs["n"]
    assert restored.clock == 30.0
    assert state.current_key == "n@a#0"
    assert state.elapsed == 30.0
    assert state.position == (1.0, 1.0)
    assert state.status == ACTIVE

    # 继续推进：进度累加而不是从日程起点重来
    restored.advance(10)
    assert restored.npcs["n"].elapsed == 40.0
    assert restored.npcs["n"].current_key == "n@a#0"


def test_load_does_not_replay_past_events():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule(
            "n",
            [
                ScheduleEvent("a", "n", 0, 60, "tavern", (1.0, 1.0)),
                ScheduleEvent("b", "n", 60, 60, "street", (2.0, 2.0)),
            ],
        )
    )
    world.skip(90, budget=1)
    snapshot = copy.deepcopy(world.snapshot())

    restored = ScheduleWorld()
    load(restored, snapshot)
    before = dict(restored.metrics)
    restored.advance(1)
    restored.run_catchup()
    # 已处理事件不会被再次派发
    assert restored.metrics["dispatched"] == before["dispatched"]


def test_snapshot_is_json_safe_and_roundtrips():
    world = ScheduleWorld()
    world.set_schedule(_day_of_events(count=4))
    world.skip(75, budget=4)
    payload = json.dumps(world.snapshot())          # 可序列化

    restored = ScheduleWorld()
    restored.restore(json.loads(payload))
    assert restored.snapshot()["clock"] == world.clock
    assert set(restored.npcs) == set(world.npcs)


# ---------------------------------------------------------------------------
# 5. 替补有恢复条件
# ---------------------------------------------------------------------------

def _weather_world():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule(
            "n",
            [
                ScheduleEvent("park", "n", 0, 240, "park", (9.0, 9.0)),
                ScheduleEvent(
                    "shelter", "n", 0, 240, "library", (2.0, 2.0),
                    indoor=True, substitutes_for="park",
                ),
                ScheduleEvent("dinner", "n", 240, 60, "tavern", (3.0, 3.0), seat="chair"),
            ],
        )
    )
    return world


def test_rain_switches_outdoor_to_indoor_substitute():
    world = _weather_world()
    world.advance(1)
    world.run_catchup()
    assert world.npcs["n"].current_key == "n@park#0"

    world.set_weather("rain")
    state = world.npcs["n"]
    assert state.status == SUBSTITUTE
    assert state.current_key == "n@shelter#0"
    assert state.position == (2.0, 2.0)
    assert state.substitute_for == "n@park#0"


def test_clear_weather_restores_original_outdoor_event():
    world = _weather_world()
    world.advance(1)
    world.run_catchup()
    world.set_weather("rain")

    report = world.recover_substitutes()
    assert report["recovered_count"] == 0          # 还在下雨，不恢复

    world.set_weather("clear")
    report = world.recover_substitutes()

    state = world.npcs["n"]
    assert report["recovered_count"] == 1
    assert state.status == ACTIVE
    assert state.current_key == "n@park#0"
    assert state.position == (9.0, 9.0)
    assert state.substitute_for is None


def test_expired_original_settles_to_current_schedule_not_stuck():
    world = _weather_world()
    world.advance(1)
    world.run_catchup()
    world.set_weather("rain")

    world.advance(269)                              # 原日程 0-240 已过期，dinner 进行中
    world.run_catchup()                             # 补算批次处理 dinner
    world.set_weather("clear")
    report = world.recover_substitutes()

    state = world.npcs["n"]
    # 关键：不得永远停在 substitute 状态，且应落到当前时刻真正生效的日程。
    assert state.status != SUBSTITUTE
    assert state.substitute_for is None
    assert state.current_key == "n@dinner#240"
    assert state.position == (3.0, 3.0)
    assert world.occupancy.owner("chair") == "n"
    # 恢复流程必须给出结论（恢复或过期结算），不能静默卡住。
    assert report["recovered_count"] + report["expired_count"] >= 1


def test_substitute_relation_ends_with_original_window():
    """原户外日程结束后，替补关系必须失效，NPC 回到正常时间线。"""
    world = ScheduleWorld()
    world.set_schedule(
        Schedule(
            "n",
            [
                ScheduleEvent("park", "n", 0, 60, "park", (9.0, 9.0)),
                ScheduleEvent(
                    "shelter", "n", 0, 60, "library", (2.0, 2.0),
                    indoor=True, substitutes_for="park",
                ),
                ScheduleEvent("tea", "n", 60, 60, "library", (2.5, 2.5), indoor=True),
            ],
        )
    )
    world.advance(1)
    world.run_catchup()
    world.set_weather("rain")
    assert world.npcs["n"].status == SUBSTITUTE

    world.advance(70)                               # park/shelter 窗口结束，tea 进行中
    world.run_catchup()                             # 补算处理 tea
    state = world.npcs["n"]
    assert state.status != SUBSTITUTE
    assert state.substitute_for is None
    assert state.current_key == "n@tea#60"
    assert state.position == (2.5, 2.5)


def test_substitute_never_jumps_into_a_future_indoor_event():
    """0-60 的户外日程不能被 120-180 的室内事件顶替（时段不重叠）。"""
    world = ScheduleWorld()
    world.set_schedule(
        Schedule(
            "n",
            [
                ScheduleEvent("park", "n", 0, 60, "park", (9.0, 9.0)),
                ScheduleEvent("future", "n", 120, 60, "library", (2.0, 2.0), indoor=True),
            ],
        )
    )
    world.advance(1)
    world.run_catchup()
    world.set_weather("rain")

    state = world.npcs["n"]
    assert state.current_key != "n@future#120"
    assert state.position != (2.0, 2.0)
    assert state.status == WAITING_WEATHER


def test_future_indoor_event_not_selected_as_substitute_at_catchup():
    """补算时同样不能把 NPC 丢进未来的室内日程。"""
    world = ScheduleWorld()
    world.set_schedule(
        Schedule(
            "n",
            [
                ScheduleEvent("park", "n", 0, 60, "park", (9.0, 9.0)),
                ScheduleEvent("future", "n", 120, 60, "library", (2.0, 2.0), indoor=True),
            ],
        )
    )
    world.set_weather("rain")
    world.skip(10, budget=4)

    state = world.npcs["n"]
    assert state.current_key is None
    assert state.position is None
    assert state.status == WAITING_WEATHER


def test_outdoor_without_substitute_waits_then_resumes():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("n", [ScheduleEvent("park", "n", 0, 120, "park", (9.0, 9.0))])
    )
    world.advance(1)
    world.run_catchup()
    world.set_weather("rain")
    assert world.npcs["n"].status == WAITING_WEATHER
    assert world.npcs["n"].current_key is None

    world.set_weather("clear")
    report = world.recover_substitutes()
    assert report["recovered_count"] == 1
    assert world.npcs["n"].status == ACTIVE
    assert world.npcs["n"].current_key == "n@park#0"


def test_rain_during_catchup_never_executes_outdoor_event():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule(
            "n",
            [
                ScheduleEvent("park", "n", 0, 120, "park", (9.0, 9.0)),
                ScheduleEvent(
                    "shelter", "n", 0, 120, "library", (2.0, 2.0),
                    indoor=True, substitutes_for="park",
                ),
            ],
        )
    )
    world.set_weather("rain")
    world.skip(10, budget=4)

    state = world.npcs["n"]
    assert state.current_key == "n@shelter#0"
    assert state.position == (2.0, 2.0)
    assert state.substitute_for == "n@park#0"


def test_substitute_recover_entry_runs_demo_cycle():
    report = substitute_recover(1)
    assert report is not None
    assert "recovered" in report


# ---------------------------------------------------------------------------
# 6. 补算事件去重
# ---------------------------------------------------------------------------

def test_repeated_catchup_does_not_dispatch_same_event_twice():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("n", [ScheduleEvent("a", "n", 0, 60, "tavern", (1.0, 1.0))])
    )
    world.skip(10, budget=4)
    first = world.metrics["dispatched"]

    # 同一批补算再跑一次：队列已空，不会重复派发
    again = world.run_catchup()
    assert again["total"] == 0
    assert world.metrics["dispatched"] == first

    # 直接重复投递同一条事件（模拟重叠窗口 / 重放）也只会入队一次
    event = world.schedules["n"].events[0]
    assert world.queue_event(event) is False
    assert world.metrics["deduped"] >= 1


def test_queued_and_deferred_are_deduped():
    world = ScheduleWorld()
    event = ScheduleEvent("walk", "n", 0, 120, "meadow", (9.0, 9.0))
    world.set_schedule(Schedule("n", [event]))
    world.regions.mark_loading("meadow")

    world.advance(1)
    assert len(world.pending) == 1
    assert world.queue_event(event) is False        # 已在待办队列
    world.run_catchup()
    assert len(world.pending) == 0
    assert len(world.deferred) == 1
    assert world.queue_event(event) is False        # 已挂起

    world.regions.mark_ready("meadow")
    world.retry_deferred()
    assert world.queue_event(event) is False        # 已执行
    assert world.metrics["dispatched"] == 1


def test_dispatch_dedupes_by_stable_key():
    world = ScheduleWorld()
    event = ScheduleEvent("a", "n", 0, 60, "tavern", (1.0, 1.0))
    world.set_schedule(Schedule("n", [event]))
    world.advance(1)

    first = dispatch(world, event)
    second = dispatch(world, event)

    assert first["executed"] is True
    assert second["duplicate"] is True
    assert second["executed"] is False


def test_same_event_dispatched_from_formats_is_one_dispatch():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("n", [ScheduleEvent("a", "n", 0, 60, "tavern", (1.0, 1.0))])
    )
    world.advance(1)
    event = world.schedules["n"].events[0]

    dispatch(world, event)
    dispatch(world, event.to_dict())                # 同 key 的 dict 形态
    assert world.metrics["dispatched"] == 1


def test_catchup_entry_is_deterministic():
    def build():
        world = ScheduleWorld()
        world.set_schedule(_day_of_events(count=8))
        world.advance(150)
        return world

    a = build()
    b = build()
    ra = catchup(a, 3)
    rb = catchup(b, 3)
    assert ra["batches"] == rb["batches"]
    assert a.snapshot()["npcs"] == b.snapshot()["npcs"]


def test_determinism_entry_reports_stable_result():
    world = ScheduleWorld()
    world.set_schedule(_day_of_events(count=6))
    report = determinism(world, runs=3)
    assert report["deterministic"] is True
    assert report["runs"] == 3


# ---------------------------------------------------------------------------
# 其他：校验、兜底、并发安全
# ---------------------------------------------------------------------------

def test_validate_detects_duplicate_events():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("n", [ScheduleEvent("a", "n", 0, 60), ScheduleEvent("a", "n", 0, 60)])
    )
    report = validate_schedule(world)
    assert report["ok"] is False
    assert any("duplicate" in error for error in report["errors"])


def test_fallback_holds_position_without_teleporting():
    world = ScheduleWorld()
    world.add_npc("n", position=(5.0, 6.0))
    report = fallback(world, "n")
    assert report["position"] == (5.0, 6.0)
    assert report["teleported"] is False


def test_metrics_reports_conflicts_and_batches():
    world = ScheduleWorld()
    world.set_schedule(
        Schedule("alice", [ScheduleEvent("sit", "alice", 0, 60, "t", (1.0, 1.0), seat="chair")])
    )
    world.set_schedule(
        Schedule("bob", [ScheduleEvent("sit", "bob", 0, 60, "t", (1.0, 2.0), seat="chair")])
    )
    world.advance(1)
    world.run_catchup()
    report = metrics(world)
    assert report["seat_conflicts"] >= 1
    assert report["batches"] >= 1


def test_concurrent_seat_reservations_have_one_winner():
    occ = OccupancyRegistry()
    results = []
    start = threading.Barrier(8)

    def worker(index):
        start.wait()
        results.append(occ.reserve("chair", f"npc-{index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [r for r in results if r.ok]
    assert len(winners) == 1
    assert occ.owner("chair") == winners[0].owner
    assert occ.conflicts == 7
