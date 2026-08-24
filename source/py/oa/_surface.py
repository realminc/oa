"""Install OA's canonical public Python surface.

The private ``_oa`` extension owns native registrations. This module only
publishes those registrations under their stable public owners:

* domain modules are lowercase: ``oa.core``, ``oa.ml``, ``oa.vision``;
* stateless operation namespaces keep their C++ spelling: ``oa.FnMatrix``;
* classes and value types are also available from the package root;
* no ``Oa*`` compatibility aliases or duplicate operation routes are created.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any
import sys

from ._schemaSurface import SCHEMA_ROOT_EXPORTS


_DOMAIN_KEYS = (
    "audio",
    "core",
    "crypto",
    "ml",
    "plot",
    "runtime",
    "ui",
    "vision",
)

_FUNCTION_KEYS = (
    "FnAudio",
    "FnAdvantage",
    "FnAutograd",
    "FnDetection",
    "FnEnvironment",
    "FnHash",
    "FnImage",
    "FnLoss",
    "FnMatrix",
    "FnMetric",
    "FnPolicy",
)


def _publicNames(module: ModuleType) -> tuple[str, ...]:
    return tuple(name for name in dir(module) if not name.startswith("_"))


def _publish(package: dict[str, Any], name: str, value: Any, owner: str) -> None:
    previous = package.get(name)
    if previous is not None and previous is not value:
        raise ImportError(
            f"OA Python symbol collision for {name}: {owner} does not match "
            "the previously registered object"
        )
    package[name] = value


def _installModule(
    package: dict[str, Any], publicName: str, module: ModuleType
) -> None:
    qualifiedName = f"oa.{publicName}"
    try:
        module.__name__ = qualifiedName
        module.__package__ = "oa"
    except (AttributeError, TypeError):
        pass
    sys.modules[qualifiedName] = module
    _publish(package, publicName, module, module.__name__)


def installSurface(
    package: dict[str, Any], sources: dict[str, ModuleType]
) -> tuple[str, ...]:
    """Publish the native modules and return the root wildcard inventory."""

    publicNames: list[str] = []

    for key in _DOMAIN_KEYS:
        source = sources.get(key)
        if source is None:
            continue
        if key == "crypto" and not bool(getattr(source, "available", True)):
            continue
        _installModule(package, key, source)
        publicNames.append(key)
        for name in _publicNames(source):
            value = getattr(source, name)
            if isinstance(value, type):
                try:
                    value.__module__ = "oa.plot" if key == "plot" else "oa"
                except (AttributeError, TypeError):
                    pass
            _publish(package, name, value, source.__name__)
            publicNames.append(name)

    for key in _FUNCTION_KEYS:
        source = sources.get(key)
        if source is None:
            continue
        _installModule(package, key, source)
        publicNames.append(key)

    # Structured operation results and FnMatrix configuration/value types are
    # useful root values, but operation functions remain owned by FnMatrix.
    fnMatrix = sources.get("FnMatrix")
    if fnMatrix is not None:
        resultNames = {
            publicName
            for publicName, _ in SCHEMA_ROOT_EXPORTS.get("FnMatrix", ())
        }
        resultNames.update({
            "LinearWeightBiasBwdResult",
            "Mamba3PreprocessConfig",
            "Quantization",
            "QuantMatrix",
            "SsmConfig",
        })
        for name in sorted(resultNames):
            if not hasattr(fnMatrix, name):
                raise ImportError(
                    f"OA Python FnMatrix registration is missing {name}"
                )
            _publish(package, name, getattr(fnMatrix, name), fnMatrix.__name__)
            publicNames.append(name)

    return tuple(sorted(set(publicNames)))


__all__ = ["installSurface"]
