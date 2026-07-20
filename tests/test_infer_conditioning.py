from scripts.infer.run_3371_full8000_infer_filter_recon import condition_feature_dim


def test_condition_feature_dim_validates_domain_layout() -> None:
    assert condition_feature_dim(condition_dim=10, domain_count=2, domain_index=1) == 8

    for kwargs in (
        {"condition_dim": 10, "domain_count": 0, "domain_index": 0},
        {"condition_dim": 10, "domain_count": 2, "domain_index": 2},
        {"condition_dim": 2, "domain_count": 2, "domain_index": 0},
    ):
        try:
            condition_feature_dim(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid condition layout was accepted: {kwargs}")
