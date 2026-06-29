from app.classifiers.base import BaseClassifier
from app.classifiers.rule_classifier import RuleBasedClassifier


class ClassifierFactory:
    """Instantiate the appropriate classifier.

    Currently the only supported classifier is the deterministic
    RuleBasedClassifier. This factory provides an extension point for
    future classifier implementations without changing call sites.
    """

    @staticmethod
    def get_classifier() -> BaseClassifier:
        """Return a ready-to-use document classifier instance.

        Returns:
            A concrete BaseClassifier implementation.
        """
        return RuleBasedClassifier()
