"""Shared spaCy NLP pipeline singleton.

Loads ``en_core_web_sm`` exactly once per process and reuses it everywhere.
The EntityRuler for TECH_SKILL patterns is added to the pipeline here so
that it is available to all extractors without any extractor needing to know
about pattern management.

Graceful degradation
--------------------
If spaCy or the model package is not installed (e.g. during CI or cold-start
before the model is downloaded), ``get_nlp()`` returns ``None``.  All callers
must guard against ``None`` and fall back to rule/regex results.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("app")

# Thread-safe initialisation — only the first call does the work.
_lock = threading.Lock()
_nlp = None          # spacy.Language or None
_initialised = False  # True once we have tried (even if it failed)


def get_nlp():
    """Return the shared spaCy Language object, or None on failure.

    The pipeline is created with only the components needed for NER so that
    load time and memory footprint are minimised:
        tok2vec  → required by the NER component
        ner      → named entity recognition (PERSON, ORG, DATE, GPE, MONEY…)

    The EntityRuler is inserted BEFORE the NER component so that its patterns
    take priority over the statistical model for TECH_SKILL entities.
    """
    global _nlp, _initialised

    if _initialised:
        return _nlp

    with _lock:
        # Double-checked locking: another thread may have initialised while we
        # were waiting for the lock.
        if _initialised:
            return _nlp

        try:
            import spacy
            from app.entity_extractors.entity_ruler_patterns import get_patterns

            # Load only the components we need.
            nlp = spacy.load(
                "en_core_web_sm",
                disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
            )

            # Add the EntityRuler before NER so TECH_SKILL patterns take
            # precedence over the statistical NER model.
            ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
            ruler.add_patterns(get_patterns())

            _nlp = nlp
            logger.info(
                "spaCy pipeline loaded: model=en_core_web_sm, "
                "pipes=%s, EntityRuler patterns=%d",
                nlp.pipe_names,
                len(get_patterns()),
            )

        except OSError:
            # Model not downloaded yet.
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm. "
                "NLP-enhanced extraction will be skipped."
            )
            _nlp = None

        except ImportError:
            logger.warning(
                "spaCy is not installed. "
                "NLP-enhanced extraction will be skipped."
            )
            _nlp = None

        except Exception as exc:
            logger.error("Failed to initialise spaCy pipeline: %s", exc)
            _nlp = None

        finally:
            _initialised = True

    return _nlp
