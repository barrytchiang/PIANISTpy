Settings tables
===============

Every numerical setting of the package is a declared field of one group, with its value in each tested tier and
one line on what it controls. The tables below are generated from the package's own declaration at every build
of this manual, so they cannot lag the code. The settings container is :class:`pianistpy.Settings`; a stage reads
its group and nothing else; an override of an undeclared field, of the wrong type or not positive is refused.

.. include:: _settings_tables.rst
