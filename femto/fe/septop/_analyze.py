"""Analyze the output of SepTop calculations."""

import typing

import numpy
from openmm import openmm

import femto.fe.corrections
import femto.fe.ddg
import femto.md.constants

if typing.TYPE_CHECKING:
    import pandas

    import femto.fe.septop


def compute_complex_correction(
    config: "femto.fe.septop.SepTopPhaseConfig", system: openmm.System
) -> dict[str, float]:
    """Computes the complex phase correction to the free energy due to the extra
    restraints.

    Args:
        config: The complex phase configuration.
        system: The complex phase system.

    Returns:
        The complex phase correction.
    """
    import femto.fe.septop

    restraints = [
        force
        for force in system.getForces()
        if force.getName() == femto.md.constants.OpenMMForceName.ALIGNMENT_RESTRAINT
    ]
    assert len(restraints) in {1, 2}, "at most two alignment restraints expected."

    correction_signs = {
        femto.fe.septop.LAMBDA_BORESCH_LIGAND_1: -1.0,  # turning on
        femto.fe.septop.LAMBDA_BORESCH_LIGAND_2: 1.0,  # turning off
    }

    result = {}

    correction = 0.0 * openmm.unit.kilocalorie_per_mole

    for restraint in restraints:
        assert restraint.getNumBonds() == 1
        _, restraint_params_list = restraint.getBondParameters(0)

        restraint_params = {
            restraint.getPerBondParameterName(i): restraint_params_list[i]
            for i in range(restraint.getNumPerBondParameters())
        }

        restraint_name = restraint.getGlobalParameterName(0)
        ligand = "lig_1" if "lig_1" in restraint_name else "lig_2"

        sign = correction_signs[restraint_name]

        value = sign * femto.fe.corrections.analytical_correction_boresch(
            restraint_params["dist_0"] * openmm.unit.nanometer,
            restraint_params["theta_a_0"] * openmm.unit.radian,
            restraint_params["theta_b_0"] * openmm.unit.radian,
            config.setup.restraints,
            config.sample.temperature,
        )

        result[f"complex_ddG_correction_{ligand}_kcal_mol"] = value.value_in_unit(
            openmm.unit.kilocalorie_per_mole
        )
        correction += value

    result["complex_ddG_correction_kcal_mol"] = correction.value_in_unit(
        openmm.unit.kilocalorie_per_mole
    )

    return result


def compute_solution_correction(
    config: "femto.fe.septop.SepTopPhaseConfig", system: openmm.System
) -> dict[str, float]:
    """Computes the solution phase correction to the free energy due to the extra
    restraints.

    Args:
        config: The solution phase configuration.
        system: The solution phase system.

    Returns:
        The solution phase correction.
    """

    restraints = [
        force
        for force in system.getForces()
        if force.getName() == femto.md.constants.OpenMMForceName.ALIGNMENT_RESTRAINT
    ]
    assert len(restraints) in {0, 1}, "at most one alignment restraint expected."

    if len(restraints) == 0:
        ln_z = 0.0  # likely ABFE
    elif isinstance(restraints[0], openmm.HarmonicBondForce):
        # consistent with SepTop paper, need to think through though
        ln_z = 0.0

        restraint = restraints[0]
        assert restraint.getNumBonds() == 1, "only one alignment restraint expected."
    else:
        raise NotImplementedError()

    correction = openmm.unit.MOLAR_GAS_CONSTANT_R * config.sample.temperature * ln_z
    return {
        "solution_ddG_correction_kcal_mol": correction.value_in_unit(
            openmm.unit.kilocalorie_per_mole
        )
    }


def compute_ddg(
    config: "femto.fe.septop.SepTopConfig",
    complex_u_kn: numpy.ndarray,
    complex_n_k: numpy.ndarray,
    complex_system: openmm.System,
    solution_u_kn: numpy.ndarray,
    solution_n_k: numpy.ndarray,
    solution_system: openmm.System,
) -> "pandas.DataFrame":
    """Computes the binding free energy from the complex and solution phase samples.

    Args:
        config: The configuration.
        complex_u_kn: The complex phase samples.
        complex_n_k: The complex phase sample counts.
        complex_system: The complex phase system.
        solution_u_kn: The solution phase samples.
        solution_n_k: The solution phase sample counts.
        solution_system: The solution phase system.

    Returns:
        A pandas DataFrame containing the total binding free energy and its components.
    """
    import pandas

    import femto.fe.septop

    samples = {
        "complex": (complex_u_kn, complex_n_k),
        "solution": (solution_u_kn, solution_n_k),
    }
    results = {}

    for phase in "complex", "solution":
        phase_config: femto.fe.septop.SepTopPhaseConfig = getattr(config, phase)
        phase_u_kn, phase_n_k = samples[phase]

        estimated, _ = femto.fe.ddg.estimate_ddg(
            phase_u_kn, phase_n_k, phase_config.sample.temperature
        )
        del estimated["ddG_0_kcal_mol"]
        del estimated["ddG_0_error_kcal_mol"]

        results.update({f"{phase}_{k}": v for k, v in estimated.items()})

    results.update(compute_complex_correction(config.complex, complex_system))
    results.update(compute_solution_correction(config.solution, solution_system))

    results["ddG_kcal_mol"] = (
        results["complex_ddG_kcal_mol"] + results["complex_ddG_correction_kcal_mol"]
    ) - (results["solution_ddG_kcal_mol"] + results["solution_ddG_correction_kcal_mol"])

    results["ddG_error_kcal_mol"] = numpy.sqrt(
        results["complex_ddG_error_kcal_mol"] ** 2
        + results["solution_ddG_error_kcal_mol"] ** 2
    )

    return pandas.DataFrame([results])
