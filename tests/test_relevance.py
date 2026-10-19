"""Tests for the topic-coverage check of the quiz / exercise generators."""

from __future__ import annotations

from ela.pedagogy.relevance import covers, normalize, topic_terms

ARMA = "Processus ARMA(p,q) : stationnarité, inversibilité, fonction d'autocorrélation, mémoire longue SARMA"
GARCH = (
    "Modèle GARCH(1,1) : volatilité conditionnelle, hétéroscédasticité, estimateur du quasi-maximum de vraisemblance"
)


def test_normalize_strips_accents():
    assert normalize("Stationnarité Élevée") == "stationnarite elevee"


def test_topic_terms_keeps_acronyms_and_prefixes():
    terms = topic_terms("modèle GARCH et estimateurs de volatilité réalisée")
    assert [(t.key, t.required) for t in terms] == [
        ("GARCH", True),
        ("estima", False),
        ("volati", False),
        ("realis", False),
    ]
    assert topic_terms("les modèles") == []  # stopwords only → nothing to check
    assert [t.required for t in topic_terms("VaR")] == [True]


def test_covers_rejects_passages_off_topic():
    assert not covers("GARCH", [ARMA])
    assert not covers("les garch", [ARMA])  # lower-case: the only term must still appear
    assert not covers("modèles GARCH", [ARMA])
    assert covers("GARCH", [GARCH])
    assert covers("processus ARMA", [ARMA])
    assert covers("stationnarité", [ARMA])


def test_covers_accepts_half_of_the_terms_but_every_acronym():
    assert covers("estimateurs de volatilité réalisée", [GARCH])  # 2 of 3 terms
    assert not covers("estimateurs ARCH de volatilité réalisée", [ARMA])  # acronym missing
    assert covers("sujet totalement inconnu", [ARMA]) is False
    assert covers("les modèles", [ARMA])  # nothing to check → accept


def test_acronyms_are_case_sensitive():
    assert covers("VaR", ["backtesting de la VaR"]) and not covers("VaR", ["modèle VAR(p)"])
    assert covers("var", ["modèle VAR(p)"])  # not an acronym as typed: case-insensitive


def test_covers_all_terms():
    assert not covers("estimateurs de volatilité réalisée", [GARCH], all_terms=True)
    assert covers("estimateurs de volatilité réalisée", [GARCH + " variance réalisée"], all_terms=True)
