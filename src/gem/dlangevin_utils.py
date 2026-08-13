from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


TWO_BETA_SCALAR_PROPOSALS = frozenset(
    {
        "dlangevintwobetas",
        "dlangevin_two_betas",
        "dl_twobetas",
        "dlang_twobetas",
    }
)
TWO_BETA_VECTOR_PROPOSALS = frozenset(
    {
        "dlangevintwobetas_vec",
        "dlangevin_two_betas_vec",
        "dl_twobetas_vec",
        "dlang_twobetas_vec",
    }
)
TWO_BETA_PROPOSALS = TWO_BETA_SCALAR_PROPOSALS | TWO_BETA_VECTOR_PROPOSALS
TWO_BETA_ANNEALING_SCALAR_PROPOSALS = frozenset(
    {
        "dlangevintwobetas_annealing",
        "dl_twobetas_annealing",
        "dlang_twobetas_anneal",
    }
)
TWO_BETA_ANNEALING_VECTOR_PROPOSALS = frozenset(
    {
        "dlangevintwobetas_annealing_vec",
        "dlangevintwobetas_anneal_vec",
        "dlangevin_two_betas_annealing_vec",
        "dlangevin_two_betas_annealing_vec_no_origin",
        "dl_twobetas_annealing_vec",
        "dlang_twobetas_anneal_vec",
    }
)
TWO_BETA_ANNEALING_PROPOSALS = (
    TWO_BETA_ANNEALING_SCALAR_PROPOSALS | TWO_BETA_ANNEALING_VECTOR_PROPOSALS
)


def _maybe_get(cfg: Any, key: str) -> Any:
    """Attempt to read `key` from a Hydra/OmegaConf object or plain dict."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key, None)
    try:
        value = getattr(cfg, key)
    except AttributeError:
        try:
            value = cfg[key]  # type: ignore[index]
        except Exception:
            return None
    return value


def resolve_two_beta_kwargs(
    proposal: str,
    primary: Any,
    fallback: Any = None,
) -> Dict[str, float]:
    """Resolve fixed proposal and MH betas for a two-beta DLangevin kernel."""
    proposal_name = str(proposal or "").strip().lower()
    if proposal_name not in TWO_BETA_PROPOSALS:
        return {}

    resolved: Dict[str, float] = {}
    for key in ("dl_beta_prop", "dl_beta_mh"):
        value = _maybe_get(primary, key)
        if value is None and fallback is not None:
            value = _maybe_get(fallback, key)
        if value is None:
            raise ValueError(f"proposal='{proposal}' requires {key} to be set.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"proposal='{proposal}' requires numeric {key}; got {value!r}."
            ) from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(
                f"proposal='{proposal}' requires finite {key} > 0; got {value!r}."
            )
        resolved[key] = parsed
    return resolved


def resolve_two_beta_annealing_kwargs(
    proposal: str,
    primary: Any,
    fallback: Any = None,
) -> Dict[str, Any]:
    """Resolve a fixed proposal beta and a linearly annealed MH beta schedule."""
    proposal_name = str(proposal or "").strip().lower()
    if proposal_name not in TWO_BETA_ANNEALING_PROPOSALS:
        return {}

    resolved: Dict[str, Any] = {}
    for key in ("dl_beta_prop", "dl_beta_mh_init", "dl_beta_mh_final"):
        value = _maybe_get(primary, key)
        if value is None and fallback is not None:
            value = _maybe_get(fallback, key)
        if value is None:
            raise ValueError(f"proposal='{proposal}' requires {key} to be set.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"proposal='{proposal}' requires numeric {key}; got {value!r}."
            ) from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(
                f"proposal='{proposal}' requires finite {key} > 0; got {value!r}."
            )
        resolved[key] = parsed

    steps_key = "dl_beta_mh_anneal_steps"
    steps_value = _maybe_get(primary, steps_key)
    if steps_value is None and fallback is not None:
        steps_value = _maybe_get(fallback, steps_key)
    if steps_value is None:
        raise ValueError(f"proposal='{proposal}' requires {steps_key} to be set.")
    try:
        steps = int(steps_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"proposal='{proposal}' requires integer {steps_key}; got {steps_value!r}."
        ) from exc
    if steps < 0:
        raise ValueError(
            f"proposal='{proposal}' requires {steps_key} >= 0; got {steps_value!r}."
        )
    resolved[steps_key] = steps
    return resolved


def resolve_dl_parameters(
    primary: Any,
    fallback: Any = None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Resolve DLangevin base parameters and optional dual schedule from config sections.

    Parameters
    ----------
    primary : Any
        Main config section (e.g. cfg.train, cfg.sample, cfg.animation).
    fallback : Any, optional
        Secondary config consulted when a key is missing in `primary`.

    Returns
    -------
    (base_beta, base_lambda_X, base_lambda_E, dual_kwargs)
        Float base parameters suitable for single-parameter DLangevin usage and a
        dictionary of dual-parameter arguments (possibly empty) that can be
        forwarded to `sampler.mcmc_sample_batch`.
    """

    def _pick(key: str) -> Optional[float]:
        value = _maybe_get(primary, key)
        if value is None and fallback is not None:
            value = _maybe_get(fallback, key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    energy_threshold_raw = _maybe_get(primary, "energy_threshold")
    if energy_threshold_raw is None and fallback is not None:
        energy_threshold_raw = _maybe_get(fallback, "energy_threshold")
    if energy_threshold_raw is False or energy_threshold_raw is None:
        energy_threshold = None
    elif isinstance(energy_threshold_raw, str) and energy_threshold_raw.strip().lower() in {
        "",
        "false",
        "none",
        "null",
    }:
        energy_threshold = None
    else:
        try:
            energy_threshold = float(energy_threshold_raw)
        except (TypeError, ValueError):
            energy_threshold = None

    beta_single = _pick("dl_beta")
    beta_init = _pick("dl_beta_init")
    beta_final = _pick("dl_beta_final")
    lambda_X_single = _pick("dl_lambda_X")
    lambda_E_single = _pick("dl_lambda_E")

    beta_near = _pick("dl_beta_near")
    lambda_X_near = _pick("dl_lambda_X_near")
    lambda_E_near = _pick("dl_lambda_E_near")

    beta_far = _pick("dl_beta_far")
    lambda_X_far = _pick("dl_lambda_X_far")
    lambda_E_far = _pick("dl_lambda_E_far")

    base_beta = (
        beta_single
        if beta_single is not None
        else (beta_far if beta_far is not None else (beta_near if beta_near is not None else 1.0))
    )
    if beta_single is None and beta_far is None and beta_near is None:
        if beta_final is not None:
            base_beta = float(beta_final)
        elif beta_init is not None:
            base_beta = float(beta_init)
    base_lambda_X = (
        lambda_X_single
        if lambda_X_single is not None
        else (
            lambda_X_far if lambda_X_far is not None else (lambda_X_near if lambda_X_near is not None else 1.0)
        )
    )
    base_lambda_E = (
        lambda_E_single
        if lambda_E_single is not None
        else (
            lambda_E_far if lambda_E_far is not None else (lambda_E_near if lambda_E_near is not None else 1.0)
        )
    )

    has_dual = (
        energy_threshold is not None
        and beta_near is not None
        and lambda_X_near is not None
        and lambda_E_near is not None
        and beta_far is not None
        and lambda_X_far is not None
        and lambda_E_far is not None
    )

    dual_kwargs: Dict[str, Any] = {}
    if has_dual:
        dual_kwargs = dict(
            energy_split_threshold=float(energy_threshold),
            dl_beta_near=float(beta_near),
            dl_lambda_X_near=float(lambda_X_near),
            dl_lambda_E_near=float(lambda_E_near),
            dl_beta_far=float(beta_far),
            dl_lambda_X_far=float(lambda_X_far),
            dl_lambda_E_far=float(lambda_E_far),
        )

    return float(base_beta), float(base_lambda_X), float(base_lambda_E), dual_kwargs


@dataclass
class ChainWarmupParams:
    enabled: bool = False
    steps: int = 0
    proposal: str = ""
    vectorized: bool = True
    gwd_beta: float = 1.0
    dl_beta: float = 1.0
    dl_lambda_X: float = 1.0
    dl_lambda_E: float = 1.0
    simple_n_edits: Optional[int] = None
    property_lambda_prop: Optional[float] = None
    dual_kwargs: Dict[str, Any] = field(default_factory=dict)


def resolve_chain_warmup(
    chain_cfg: Any,
    fallback: Any = None,
    default_gwd_beta: float = 1.0,
) -> ChainWarmupParams:
    """
    Resolve optional chain warmup configuration.

    Parameters
    ----------
    chain_cfg : Any
        Section containing `steps`, `proposal`, etc. (e.g., cfg.train.chain_warmup).
    fallback : Any, optional
        Parent section used to fill in missing DLangevin parameters.
    default_gwd_beta : float
        Used if neither warmup nor fallback specify gwd_beta.
    """

    try:
        default_gwd = float(default_gwd_beta)
    except (TypeError, ValueError):
        default_gwd = 1.0

    base = ChainWarmupParams(gwd_beta=default_gwd)
    if chain_cfg is None:
        return base

    steps_raw = _maybe_get(chain_cfg, "steps")
    try:
        steps = int(steps_raw if steps_raw is not None else 0)
    except (TypeError, ValueError):
        print(f"[warn] Invalid chain_warmup.steps={steps_raw!r}; disabling warmup.")
        steps = 0

    proposal_raw = _maybe_get(chain_cfg, "proposal") or ""
    proposal = str(proposal_raw).strip().lower()

    enabled_flag = _maybe_get(chain_cfg, "enabled")
    if enabled_flag is None:
        enabled = steps > 0
    else:
        enabled = bool(enabled_flag)

    if not enabled or steps <= 0 or not proposal:
        return base

    vectorized_raw = _maybe_get(chain_cfg, "vectorized")
    vectorized = True if vectorized_raw is None else bool(vectorized_raw)

    simple_raw = _maybe_get(chain_cfg, "simple_n_edits")
    simple_n_edits: Optional[int]
    if simple_raw is None:
        simple_n_edits = None
    else:
        try:
            simple_n_edits = int(simple_raw)
        except (TypeError, ValueError):
            print(f"[warn] Invalid chain_warmup.simple_n_edits={simple_raw!r}; ignoring override.")
            simple_n_edits = None

    gwd_beta_val = _maybe_get(chain_cfg, "gwd_beta")
    if gwd_beta_val is None and fallback is not None:
        gwd_beta_val = _maybe_get(fallback, "gwd_beta")
    if gwd_beta_val is None:
        gwd_beta_val = default_gwd
    try:
        gwd_beta = float(gwd_beta_val)
    except (TypeError, ValueError):
        print(f"[warn] Invalid chain_warmup.gwd_beta={gwd_beta_val!r}; using default {default_gwd}.")
        gwd_beta = default_gwd

    dl_beta, dl_lambda_X, dl_lambda_E, dual_kwargs = resolve_dl_parameters(chain_cfg, fallback)
    if proposal not in {"dlangevin", "dlang", "dl", "dlangevin_nomh", "dlang_nomh", "dl_nomh"}:
        dual_kwargs = {}

    prop_lambda_raw = _maybe_get(chain_cfg, "lambda_property_prop")
    if prop_lambda_raw is None:
        prop_lambda_raw = _maybe_get(chain_cfg, "property_lambda_prop")
    if prop_lambda_raw is None:
        property_lambda_prop = None
    else:
        try:
            property_lambda_prop = float(prop_lambda_raw)
        except (TypeError, ValueError):
            print(
                f"[warn] Invalid chain_warmup.property_lambda_prop={prop_lambda_raw!r}; ignoring override."
            )
            property_lambda_prop = None

    return ChainWarmupParams(
        enabled=True,
        steps=int(steps),
        proposal=proposal,
        vectorized=vectorized,
        gwd_beta=gwd_beta,
        dl_beta=dl_beta,
        dl_lambda_X=dl_lambda_X,
        dl_lambda_E=dl_lambda_E,
        simple_n_edits=simple_n_edits,
        property_lambda_prop=property_lambda_prop,
        dual_kwargs=dual_kwargs,
    )
