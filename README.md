# Conformal Data Cleaning

Error Demo Branch

Rationale:

This uses code from Sebastian's dissertation experiments since we know the performance of the methods therein due to experimental results.

The demo needs to run fast and in our first approach we subsampled models (i.e. just random forest) so it could run fast enough on an untested library (that we plumbed together hopefully correctly).

This solution is far better since we just commit and use random forest (which is what we did anyways) but now we have empirical results and therefore verification of correctness.

i.e. Is theoretically "better" with no empirical verification better than theoretically "worse" with experimental verification.

MVP first, optimizations (like adding AutoGluon) later.
