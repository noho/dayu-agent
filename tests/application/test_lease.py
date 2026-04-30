"""``dayu.host.lease`` 公共原语的单元测试。

确保 ``generate_lease_id`` 与 ``LeaseExpiredError`` 的契约稳定，避免 store
层间接覆盖时漏测基本属性。
"""

from __future__ import annotations

import re

import pytest

from dayu.host.lease import LeaseExpiredError, generate_lease_id


_HEX32 = re.compile(r"^[0-9a-f]{32}$")


@pytest.mark.unit
def test_generate_lease_id_returns_lowercase_hex32() -> None:
    """``generate_lease_id`` 必须返回 32 位小写 hex 字符串。"""

    lease = generate_lease_id()
    assert isinstance(lease, str)
    assert _HEX32.match(lease) is not None


@pytest.mark.unit
def test_generate_lease_id_is_unique_across_calls() -> None:
    """连续调用必须产出互不相同的 lease_id（uuid4 概率上保证）。"""

    leases = {generate_lease_id() for _ in range(64)}
    assert len(leases) == 64


@pytest.mark.unit
def test_lease_expired_error_carries_record_and_lease_attributes() -> None:
    """``LeaseExpiredError`` 必须带上 record_id / lease_id 供日志定位。"""

    err = LeaseExpiredError(
        "lease expired",
        record_id="delivery_1",
        lease_id="lease_abc",
    )
    assert isinstance(err, RuntimeError)
    assert str(err) == "lease expired"
    assert err.record_id == "delivery_1"
    assert err.lease_id == "lease_abc"


@pytest.mark.unit
def test_lease_expired_error_accepts_none_lease_id() -> None:
    """调用方完全未带 lease 时，``lease_id`` 应允许为 ``None``。"""

    err = LeaseExpiredError(
        "missing lease",
        record_id="delivery_2",
        lease_id=None,
    )
    assert err.lease_id is None
    assert err.record_id == "delivery_2"
