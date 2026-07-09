# HMM regime model: fit, causal decode, state->label mapping.
# The trap this avoids: hmmlearn's predict (Viterbi) and predict_proba
# (forward-backward) condition every step on the whole sequence, future
# included. We run the forward recursion only, giving the filtered posterior
# P(state_t | obs up to t) — what a real-time observer could compute on day t.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .config import HMM_CONFIG, RANDOM_STATE, HMMConfig
from .features import apply_scaler, fit_scaler

LABELS=("bull", "bear", "crisis")


@dataclass
class FittedRegime:
    # an HMM plus what's needed to use it causally later: the train-only
    # scaler and the data-driven state->label map (state indices permute).
    model: GaussianHMM
    state_to_label: dict[int, str]
    scaler_mean: pd.Series
    scaler_std: pd.Series
    train_end: pd.Timestamp


def _label_states(model: GaussianHMM) -> dict[int, str]:
    # crisis first, as the highest-vol state — vol clustering is the reliable
    # discriminator, whereas "highest return" can grab a violent 2009 rebound.
    # Of the rest, the higher-return state is bull.
    mean_ret=model.means_[:, 0]
    mean_vol=model.means_[:, 1]
    crisis=int(np.argmax(mean_vol))
    rest=[s for s in range(model.n_components) if s != crisis]
    bull=max(rest, key=lambda s: mean_ret[s])
    bear=next(s for s in rest if s != bull)
    return {bull: "bull", bear: "bear", crisis: "crisis"}


def fit_hmm(
    train_features: pd.DataFrame,
    cfg: HMMConfig=HMM_CONFIG,
    random_state: int=RANDOM_STATE,
) -> FittedRegime:
    # EM is init-sensitive, so run a few seeded restarts and keep the best
    # in-sample log-likelihood. The scaler travels with the model.
    mean, std=fit_scaler(train_features)
    X=apply_scaler(train_features, mean, std).to_numpy()

    best_model: GaussianHMM | None=None
    best_ll=-np.inf
    for i in range(cfg.n_em_inits):
        model=GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            random_state=random_state + i,
        )
        try:
            model.fit(X)
            ll=model.score(X)
        except ValueError:  # degenerate init, e.g. singular covariance
            continue
        if ll > best_ll:
            best_ll, best_model=ll, model

    if best_model is None:
        raise RuntimeError("all EM initializations failed")

    return FittedRegime(
        model=best_model,
        state_to_label=_label_states(best_model),
        scaler_mean=mean,
        scaler_std=std,
        train_end=train_features.index[-1],
    )


def filtered_posteriors(fitted: FittedRegime, features: pd.DataFrame) -> pd.DataFrame:
    # forward recursion only: row t depends on rows <= t, so a single pass
    # over a growing history is causal for every date.
    model=fitted.model
    X=apply_scaler(features, fitted.scaler_mean, fitted.scaler_std).to_numpy()
    n_states=model.n_components

    log_emit=np.column_stack(
        [
            multivariate_normal.logpdf(
                X, mean=model.means_[s], cov=model.covars_[s], allow_singular=True
            )
            for s in range(n_states)
        ]
    )
    log_trans=np.log(model.transmat_ + 1e-300)

    log_alpha=np.log(model.startprob_ + 1e-300) + log_emit[0]
    out=np.empty((len(X), n_states))
    out[0]=log_alpha - logsumexp(log_alpha)
    for t in range(1, len(X)):
        log_alpha=logsumexp(log_alpha[:, None] + log_trans, axis=0) + log_emit[t]
        out[t]=log_alpha - logsumexp(log_alpha)

    probs=pd.DataFrame(np.exp(out), index=features.index)
    probs.columns=[fitted.state_to_label[s] for s in range(n_states)]
    return probs[list(LABELS)]


def filtered_labels(fitted: FittedRegime, features: pd.DataFrame) -> pd.Series:
    # causal regime label per date: argmax of the filtered posterior
    return filtered_posteriors(fitted, features).idxmax(axis=1).rename("regime")
