import numpy
import openmm
import pymbar
import pytest
from pymbar.testsystems import harmonic_oscillators

import femto.fe.config
import femto.fe.septop
import femto.md.constants
import femto.md.restraints
from femto.fe.septop._analyze import compute_ddg
from femto.md.utils.openmm import is_close


@pytest.fixture
def mock_complex_restraint() -> femto.fe.septop.SepTopComplexRestraints:
    return femto.fe.septop.SepTopComplexRestraints(
        k_distance=20.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2,
        k_angle_a=20.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2,
        k_angle_b=20.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2,
        k_dihedral_a=20.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2,
        k_dihedral_b=20.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2,
        k_dihedral_c=20.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2,
    )


@pytest.fixture
def mock_complex_system(mock_complex_restraint) -> openmm.System:
    system = openmm.System()

    for _ in range(6):
        system.addParticle(1.0)

    restraint_1 = femto.md.restraints.create_boresch_restraint(
        mock_complex_restraint,
        (0, 1, 2),
        (3, 4, 5),
        numpy.array(
            [
                [-3.0, -1.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, -1.0, 0.0],
            ]
        )
        * openmm.unit.angstrom,
    )
    restraint_1.addGlobalParameter(femto.fe.septop.LAMBDA_BORESCH_LIGAND_1, 0.0)

    restraint_2 = femto.md.restraints.create_boresch_restraint(
        mock_complex_restraint,
        (0, 1, 2),
        (3, 4, 5),
        numpy.array(
            [
                [-3.0, -1.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-1.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, -1.0, 0.0],
            ]
        )
        * openmm.unit.angstrom,
    )
    restraint_2.addGlobalParameter(femto.fe.septop.LAMBDA_BORESCH_LIGAND_2, 0.0)

    system.addForce(restraint_1)
    system.addForce(restraint_2)

    return system


@pytest.fixture
def mock_solution_system() -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    restraint = openmm.HarmonicBondForce()
    restraint.addBond(0, 1, 1.0, 2.0)
    restraint.setName(femto.md.constants.OpenMMForceName.ALIGNMENT_RESTRAINT)
    restraint.setForceGroup(femto.md.constants.OpenMMForceGroup.ALIGNMENT_RESTRAINT)
    system.addForce(restraint)
    return system


def create_mock_samples(
    offset_k: list[float, ...], k_k: list[float, ...], temperature: openmm.unit.Quantity
) -> tuple[numpy.ndarray, numpy.ndarray, openmm.unit.Quantity]:
    test_case = harmonic_oscillators.HarmonicOscillatorsTestCase(offset_k, k_k)

    _, u_kn, n_k, _ = test_case.sample([10, 10, 10, 10, 10], mode="u_kn")

    f_i = pymbar.MBAR(u_kn, n_k).f_k
    ddg = openmm.unit.MOLAR_GAS_CONSTANT_R * temperature * (f_i[-1] - f_i[0])
    ddg = ddg.in_units_of(openmm.unit.kilocalorie_per_mole)

    return u_kn, n_k, ddg


def test_compute_ddg(
    tmp_cwd, mock_complex_restraint, mock_complex_system, mock_solution_system
):
    temperature = 300.0 * openmm.unit.kelvin

    complex_u_kn, complex_n_k, expected_complex_ddg = create_mock_samples(
        [0, 1, 2, 3, 4], [1, 2, 4, 8, 16], temperature
    )
    solution_u_kn, solution_n_k, expected_solution_ddg = create_mock_samples(
        [1, 2, 3, 4, 5], [2, 4, 8, 16, 32], temperature
    )

    config = femto.fe.septop.SepTopConfig()
    config.complex.setup.restraints = mock_complex_restraint
    config.complex.sample.temperature = temperature
    config.solution.sample.temperature = temperature

    ddg = compute_ddg(
        config,
        complex_u_kn,
        complex_n_k,
        mock_complex_system,
        solution_u_kn,
        solution_n_k,
        mock_solution_system,
    )

    assert len(ddg) == 1
    assert ddg.columns.tolist() == [
        "complex_ddG_kcal_mol",
        "complex_ddG_error_kcal_mol",
        "solution_ddG_kcal_mol",
        "solution_ddG_error_kcal_mol",
        "complex_ddG_correction_lig_1_kcal_mol",
        "complex_ddG_correction_lig_2_kcal_mol",
        "complex_ddG_correction_kcal_mol",
        "solution_ddG_correction_kcal_mol",
        "ddG_kcal_mol",
        "ddG_error_kcal_mol",
    ]

    assert is_close(
        ddg["complex_ddG_kcal_mol"].values * openmm.unit.kilocalorie_per_mole,
        expected_complex_ddg,
    )
    assert is_close(
        ddg["solution_ddG_kcal_mol"].values * openmm.unit.kilocalorie_per_mole,
        expected_solution_ddg,
    )

    assert numpy.isclose(
        ddg["ddG_kcal_mol"].values,
        (
            ddg["complex_ddG_kcal_mol"].values
            + ddg["complex_ddG_correction_kcal_mol"].values
        )
        - (
            ddg["solution_ddG_kcal_mol"].values
            + ddg["solution_ddG_correction_kcal_mol"].values
        ),
    )

    assert numpy.isclose(ddg["solution_ddG_correction_kcal_mol"].values, 0.0)

    assert not numpy.isclose(ddg["complex_ddG_correction_kcal_mol"].values, 0.0)
    assert numpy.isclose(
        ddg["complex_ddG_correction_kcal_mol"].values,
        ddg["complex_ddG_correction_lig_1_kcal_mol"].values
        + ddg["complex_ddG_correction_lig_2_kcal_mol"].values,
    )
