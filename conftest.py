"""Settings for every test run of the repository."""

import os

# Cache off unless set: eight runs at once on one bounded cache folder queued
# behind each other's eviction scans (all past 50 min; off, under 4 min each).
os.environ.setdefault("PIANISTPY_COMPILATION_CACHE", "off")
