# conftest.py —— pytest 全局配置
# 注册 integration 标记：集成测试调真实 LLM/Embedding API，默认跳过，用 -m integration 开启

import pytest


def pytest_configure(config):
    """注册自定义标记，避免 warning"""
    config.addinivalue_line(
        "markers",
        "integration: 调真实 LLM/Embedding API 的测试（默认跳过，用 -m integration 运行）"
    )


def pytest_collection_modifyitems(config, items):
    """默认跳过 integration 标记的测试（除非显式用 -m integration 运行）"""
    if config.getoption("-m") == "integration":
        return  # 用户显式要求跑集成测试，不跳过
    skip_integration = pytest.mark.skip(reason="集成测试需真实 API，用 -m integration 运行")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
