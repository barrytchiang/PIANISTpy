Numerical settings
==================

Every numerical setting of the package is a field of :class:`pianistpy.Settings`. The tables below list each
field, its value under the default settings and under the quick settings (:doc:`Accuracy <../guide/accuracy>`),
and what the field controls. The fields fall in five groups, one table each: ``inversion``, ``truncation``,
``sampling``, ``observables`` and ``losvd``.

An override of a field, such as ``pp.Settings(inversion={"n_nodes": 256})``, raises
:class:`~pianistpy.InputError` when the group or the field's name is unknown. An override also
raises :class:`~pianistpy.InputError` when the value has the wrong type, or when the value is not positive for a
field that must be positive.

.. include:: _settings_tables.rst
