def test_cheapest_tier_is_first(policies):
    assert policies.cheapest_tier() is policies.tiers[0]


class TestSideChannel:
    def test_detects_title_generator(self, policies):
        assert policies.is_side_channel(
            "You are a title generator. You output ONLY a thread title."
        )

    def test_case_insensitive(self, policies):
        assert policies.is_side_channel("you ARE a Title Generator, etc")

    def test_normal_coding_system_is_not_side_channel(self, policies):
        assert not policies.is_side_channel("You are a coding agent.")

    def test_empty_system_is_not_side_channel(self, policies):
        assert not policies.is_side_channel("")
