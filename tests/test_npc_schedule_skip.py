import pytest
import gameplay.npc_schedule_skip as mod

CASES = [
    ("batch_catchup", (1,)),
    ("wait_region", (1,)),
    ("reserve_token", (1,)),
    ("save_time_offset", (1,)),
    ("substitute_recover", (1,)),
    ("advance", (1,)),
    ("catchup", (1,)),
    ("dispatch", (1,)),
    ("occupy", (1,)),
    ("load", (1,)),
    ("validate_schedule", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
