"""The map wrappers take every keyword argument of the functions they wrap.

``metacouplingllm.visualization`` exposes the map functions of
``visualization.worldmap`` and ``visualization.adm1_map`` through wrappers
that import the implementation only when called, and the package top level
wraps them again with ``**kwargs``.  A keyword missing from a wrapper's
signature raises ``TypeError`` on that import path only.
"""

from __future__ import annotations

import inspect

import pytest

import metacouplingllm
from metacouplingllm import visualization
from metacouplingllm.visualization import adm1_map, worldmap

IMPLEMENTATIONS = [
    pytest.param(module, name, id=name)
    for module, name in [
        (worldmap, "plot_focal_country_map"),
        (worldmap, "plot_analysis_map"),
        (adm1_map, "plot_focal_adm1_map"),
    ]
]


@pytest.mark.parametrize("module, name", IMPLEMENTATIONS)
def test_signature_matches_implementation(module, name):
    """Same parameters, kinds, defaults and annotations, in the same order."""
    wrapper = inspect.signature(getattr(visualization, name))
    implementation = inspect.signature(getattr(module, name))
    assert list(wrapper.parameters.values()) == list(
        implementation.parameters.values()
    )
    assert wrapper.return_annotation == implementation.return_annotation


@pytest.mark.parametrize(
    "package",
    [
        pytest.param(visualization, id="visualization"),
        pytest.param(metacouplingllm, id="top_level"),
    ],
)
@pytest.mark.parametrize("module, name", IMPLEMENTATIONS)
def test_every_keyword_is_forwarded(monkeypatch, package, module, name):
    """Each keyword reaches the implementation as the object passed in."""
    parameters = inspect.signature(getattr(module, name)).parameters
    keywords = {
        keyword: object()
        for keyword, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    calls: list[tuple[tuple, dict]] = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return "figure"

    monkeypatch.setattr(module, name, record)
    first = object()
    assert getattr(package, name)(first, **keywords) == "figure"

    (args, kwargs), = calls
    assert args == (first,)
    assert kwargs.keys() == keywords.keys()
    for keyword, value in keywords.items():
        assert kwargs[keyword] is value, keyword
