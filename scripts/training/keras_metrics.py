# -*- coding: utf-8 -*-

"""Custom metrics."""

from typing import Any, Dict, Union
import tensorflow as tf
from tensorflow import keras


class ExponentialMovingAvg(keras.metrics.Metric):
    """Exponential moving average metric."""

    def __init__(self, decay: float, **kwargs) -> None:
        """Create ExponentialMovingAvg.

        Parameters
        ----------
        decay: float
            Decay
        **kwargs
            keras.metrics.Metric args
        """
        super().__init__(**kwargs)
        self.decay = decay
        self._value = self.add_weight(
            name="value",
            initializer='zeros',
            aggregation=tf.VariableAggregation.MEAN,
            synchronization=tf.VariableSynchronization.ON_WRITE,
        )

    def update_state(self, value: tf.Tensor) -> Union[None, tf.Operation]:
        """Update metric state.

        Parameters
        ----------
        value: tf.Tensor
            Value

        Returns
        -------
        Union[None, tf.Operation]
            Update op
        """
        return self._value.assign_add(
            (1 - self.decay) * (value - self._value)
        )

    def result(self) -> tf.Tensor:
        """Get metric value.

        Returns
        -------
        tf.Tensor
            Result
        """
        return tf.identity(self._value)

    def reset_state(self) -> Union[None, tf.Operation]:
        """Reset metric state.

        Returns
        -------
        Union[None, tf.Operation]
            Update op
        """
        return self._value.assign(0)

    def get_config(self) -> Dict[str, Any]:
        """Get metric config.

        Returns
        -------
        Dict[str, Any]
            Metric config
        """
        config = super().get_config()
        return {
            **config,
            "decay": self.decay,
        }


class CounterMetric(keras.metrics.Metric):
    """Counter."""

    def __init__(self, **kwargs) -> None:
        """Create CounterMetric.

        Parameters
        ----------
        **kwargs
            keras.metrics.Metric args
        """
        super().__init__(**kwargs)
        self._value = self.add_weight(
            name="value",
            dtype=tf.int64,
            initializer='zeros',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            synchronization=tf.VariableSynchronization.NONE,
        )

    def update_state(self) -> Union[None, tf.Operation]:
        """Update metric state.

        Returns
        -------
        Union[None, tf.Operation]
            Update op
        """
        return self._value.assign_add(1)

    def result(self) -> tf.Tensor:
        """Get metric value.

        Returns
        -------
        tf.Tensor
            Result
        """
        return tf.identity(self._value)

    def reset_state(self) -> Union[None, tf.Operation]:
        """Reset metric state.

        Returns
        -------
        Union[None, tf.Operation]
            Update op
        """
        return self._value.assign(0)


CUSTOM_METRICS = {
    "ExponentialMovingAvg": ExponentialMovingAvg,
    "CounterMetric": CounterMetric,
}

__all__ = ["CUSTOM_METRICS"] + list(CUSTOM_METRICS)
