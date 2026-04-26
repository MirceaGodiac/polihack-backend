from apps.api.app import main


class FakeWindowsSelectorEventLoopPolicy:
    pass


def test_windows_event_loop_policy_is_set_for_windows(monkeypatch):
    policies = []
    monkeypatch.setattr(
        main.asyncio,
        "WindowsSelectorEventLoopPolicy",
        FakeWindowsSelectorEventLoopPolicy,
        raising=False,
    )

    configured = main._configure_windows_event_loop_policy(
        platform="win32",
        set_policy=policies.append,
    )

    assert configured is True
    assert len(policies) == 1
    assert isinstance(policies[0], FakeWindowsSelectorEventLoopPolicy)


def test_windows_event_loop_policy_is_not_set_for_non_windows(monkeypatch):
    policies = []
    monkeypatch.setattr(
        main.asyncio,
        "WindowsSelectorEventLoopPolicy",
        FakeWindowsSelectorEventLoopPolicy,
        raising=False,
    )

    configured = main._configure_windows_event_loop_policy(
        platform="linux",
        set_policy=policies.append,
    )

    assert configured is False
    assert policies == []
