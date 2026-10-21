r"""The log-likelihood of a model against measured data.

A user stores each set of measurements in a `Data` and calls `likelihood` on
a model and the data. `likelihood` computes the projected tables of the
model once and returns a `Likelihood`, a JAX function of the component
weights, of the mass and length scales and, for the Method II components
chosen as free, of the anisotropy parameters. `log_likelihood` returns the value of the
log-likelihood at the model's own weights and scales.

For measured values :math:`d`, model values :math:`m` and the covariance
matrix :math:`C` of the errors (diagonal, with :math:`C_{ii} = e_i^2`, for
independent errors :math:`e_i`), the log-likelihood of one `Data` is

.. math::

    \ln \mathcal{L} = -\frac{1}{2} (d - m)^{\rm T} C^{-1} (d - m)
    - \frac{1}{2} \ln \det (2 \pi C),

and the log-likelihoods of several `Data` are summed. Notes Section T:7.4
derives the likelihood.
"""

__all__ = ["Data", "Likelihood", "likelihood", "log_likelihood"]

from .core import Data, Likelihood, likelihood, log_likelihood
