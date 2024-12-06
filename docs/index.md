--8<-- "README.md::22"

!!! warning
    From version 0.3.0 onwards, the codebase was re-written to completely remove the dependency on `parmed`,
    allowing easy use of any force field parameters in OpenFF, Amber, and OpenMM FFXML formats. This re-write also introduced
    a number of neccessary API changes. See the [migration guide for more details](https://psivant.github.io/femto/latest/migration/).

    Further, the default protocols selected for the ATM and SepTop methods are still being tested and optimized, and may
    not be optimal. It is recommended that you run a few test calculations to ensure that the results are reasonable.

--8<-- "README.md:29:"
