import copy

import numpy
import openmm
import openmm.unit
import pytest

import femto.fe.config
import femto.md.config
from femto.fe.fep import apply_fep
from femto.md.constants import OpenMMForceGroup, OpenMMPlatform
from femto.md.rest import REST_CTX_PARAM, REST_CTX_PARAM_SQRT, apply_rest
from femto.md.utils.openmm import compute_energy, is_close

TRIANGLE_COORDS = (
    numpy.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    * openmm.unit.angstrom
)

E_CHARGE = 1.602176634e-19 * openmm.unit.coulomb
EPSILON0 = (
    1e-6
    * 8.8541878128e-12
    / (openmm.unit.AVOGADRO_CONSTANT_NA * E_CHARGE**2)
    * openmm.unit.farad
    / openmm.unit.meter
)
ONE_4PI_EPS0 = 1 / (4 * numpy.pi * EPSILON0) * EPSILON0.unit * 10.0  # nm -> angstrom

# mock values for beta_m / beta_0
BM_B0 = 4.0
BM_B0_RT = numpy.sqrt(BM_B0)


E = openmm.unit.elementary_charge
KJ_MOL = openmm.unit.kilojoules_per_mole
ANGSTROM = openmm.unit.angstrom


def create_3_particle_system(
    charges: list[float], epsilons: list[float], sigmas: list[float]
):
    system = openmm.System()

    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    assert all(len(params) == 3 for params in [charges, epsilons, sigmas])

    for charge, epsilon, sigma in zip(charges, epsilons, sigmas, strict=True):
        force.addParticle(charge, sigma, epsilon)

    system.addForce(force)

    return system


def create_3_particle_intra_system(
    interaction_pairs: list[tuple[int, int, float, float, float]],
):
    system = openmm.System()

    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    for _ in range(3):
        force.addParticle(0.0, 1.0, 0.0)

    for idx_a, idx_b, charge, epsilon, sigma in interaction_pairs:
        force.addException(idx_a, idx_b, charge, sigma, epsilon)

    system.addForce(force)

    return system


# we have three pairs of interactions, a-b, a-c, b-c. here we say only a and b are
# affected by REST, so a-b interaction should be scaled by BM_B0 while a-c and b-c
# should be scaled by sqrt(BM_B0)
@pytest.mark.parametrize(
    "system",
    [
        create_3_particle_system(
            [0.0, 0.0, 0.0] * openmm.unit.elementary_charge,
            [2.0, 3.0, 4.0] * openmm.unit.kilojoules_per_mole,
            [1.0, 1.0, 1.0] * openmm.unit.angstrom,
        ),
        create_3_particle_intra_system(
            [
                (0, 1, 0.0 * E * E, numpy.sqrt(2.0 * 3.0) * KJ_MOL, 1.0 * ANGSTROM),
                (0, 2, 0.0 * E * E, numpy.sqrt(2.0 * 4.0) * KJ_MOL, 1.0 * ANGSTROM),
                (1, 2, 0.0 * E * E, numpy.sqrt(3.0 * 4.0) * KJ_MOL, 1.0 * ANGSTROM),
            ]
        ),
    ],
)
def test_epsilon_ss_and_sw(system):
    energy_0 = compute_energy(
        system, TRIANGLE_COORDS, None, {}, OpenMMPlatform.REFERENCE
    )

    expected_force_groups = [force.getForceGroup() for force in system.getForces()]
    apply_rest(system, {0, 1}, femto.md.config.REST(scale_nonbonded=True))

    force_groups = [force.getForceGroup() for force in system.getForces()]
    assert force_groups == expected_force_groups

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: 1.0, REST_CTX_PARAM_SQRT: 1.0},
        OpenMMPlatform.REFERENCE,
    )
    assert is_close(energy, energy_0)

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: 0.0, REST_CTX_PARAM_SQRT: 0.0},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = 0.0 * openmm.unit.kilojoules_per_mole
    assert is_close(energy, expected_energy)

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: BM_B0, REST_CTX_PARAM_SQRT: BM_B0_RT},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = (
        4.0
        * (
            # a - b should be scaled by BM_B0
            BM_B0 * numpy.sqrt(2.0 * 3.0) * ((1.0 / 3.0) ** 12 - (1.0 / 3.0) ** 6)
            # a - c should be scaled by sqrt(BM_B0)
            + BM_B0_RT * numpy.sqrt(2.0 * 4.0) * ((1.0 / 4.0) ** 12 - (1.0 / 4.0) ** 6)
            # b - c should be scaled by sqrt(BM_B0)
            + BM_B0_RT * numpy.sqrt(3.0 * 4.0) * ((1.0 / 5.0) ** 12 - (1.0 / 5.0) ** 6)
        )
        * openmm.unit.kilojoules_per_mole
    )
    assert is_close(energy, expected_energy)


@pytest.mark.parametrize(
    "system",
    [
        create_3_particle_system(
            [2.0, 3.0, 4.0] * openmm.unit.elementary_charge,
            [0.0, 0.0, 0.0] * openmm.unit.kilojoules_per_mole,
            [1.0, 1.0, 1.0] * openmm.unit.angstrom,
        ),
        create_3_particle_intra_system(
            [
                (0, 1, 2.0 * 3.0 * E * E, 0.0 * KJ_MOL, 1.0 * ANGSTROM),
                (0, 2, 2.0 * 4.0 * E * E, 0.0 * KJ_MOL, 1.0 * ANGSTROM),
                (1, 2, 3.0 * 4.0 * E * E, 0.0 * KJ_MOL, 1.0 * ANGSTROM),
            ]
        ),
    ],
)
def test_charge_ss_and_sw(system):
    energy_0 = compute_energy(
        system, TRIANGLE_COORDS, None, {}, OpenMMPlatform.REFERENCE
    )

    apply_rest(system, {0, 1}, femto.md.config.REST(scale_nonbonded=True))

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: 1.0, REST_CTX_PARAM_SQRT: 1.0},
        OpenMMPlatform.REFERENCE,
    )
    assert is_close(energy, energy_0)

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: 0.0, REST_CTX_PARAM_SQRT: 0.0},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = 0.0 * openmm.unit.kilojoules_per_mole
    assert is_close(energy, expected_energy)

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: BM_B0, REST_CTX_PARAM_SQRT: BM_B0_RT},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = (
        ONE_4PI_EPS0
        * (
            # a - b should be scaled by BM_B0
            BM_B0 * 2.0 * 3.0 / 3.0
            # a - c should be scaled by sqrt(BM_B0)
            + BM_B0_RT * 2.0 * 4.0 / 4.0
            # b - c should be scaled by sqrt(BM_B0)
            + BM_B0_RT * 3.0 * 4.0 / 5.0
        )
        * openmm.unit.kilojoules_per_mole
    )
    assert is_close(energy, expected_energy)


# here we say only a is affected by REST, so a-b and a-c interaction should be
# scaled by sqrt(BM_B0) while b-c should be unscaled.
@pytest.mark.parametrize(
    "system",
    [
        create_3_particle_system(
            [0.0, 0.0, 0.0] * openmm.unit.elementary_charge,
            [2.0, 3.0, 4.0] * openmm.unit.kilojoules_per_mole,
            [1.0, 1.0, 1.0] * openmm.unit.angstrom,
        ),
        create_3_particle_intra_system(
            [
                (0, 1, 0.0 * E * E, numpy.sqrt(2.0 * 3.0) * KJ_MOL, 1.0 * ANGSTROM),
                (0, 2, 0.0 * E * E, numpy.sqrt(2.0 * 4.0) * KJ_MOL, 1.0 * ANGSTROM),
                (1, 2, 0.0 * E * E, numpy.sqrt(3.0 * 4.0) * KJ_MOL, 1.0 * ANGSTROM),
            ]
        ),
    ],
)
def test_intermolecular_epsilon_sw_and_ww(system):
    apply_rest(system, {0}, femto.md.config.REST(scale_nonbonded=True))

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: BM_B0, REST_CTX_PARAM_SQRT: BM_B0_RT},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = (
        4.0
        * (
            # a - b should be scaled by sqrt(BM_B0)
            BM_B0_RT * numpy.sqrt(2.0 * 3.0) * ((1.0 / 3.0) ** 12 - (1.0 / 3.0) ** 6)
            # a - c should be scaled by sqrt(BM_B0)
            + BM_B0_RT * numpy.sqrt(2.0 * 4.0) * ((1.0 / 4.0) ** 12 - (1.0 / 4.0) ** 6)
            # b - c should not be scaled
            + numpy.sqrt(3.0 * 4.0) * ((1.0 / 5.0) ** 12 - (1.0 / 5.0) ** 6)
        )
        * openmm.unit.kilojoules_per_mole
    )
    assert is_close(energy, expected_energy)


@pytest.mark.parametrize(
    "system",
    [
        create_3_particle_system(
            [2.0, 3.0, 4.0] * openmm.unit.elementary_charge,
            [0.0, 0.0, 0.0] * openmm.unit.kilojoules_per_mole,
            [1.0, 1.0, 1.0] * openmm.unit.angstrom,
        ),
        create_3_particle_intra_system(
            [
                (0, 1, 2.0 * 3.0 * E * E, 0.0 * KJ_MOL, 1.0 * ANGSTROM),
                (0, 2, 2.0 * 4.0 * E * E, 0.0 * KJ_MOL, 1.0 * ANGSTROM),
                (1, 2, 3.0 * 4.0 * E * E, 0.0 * KJ_MOL, 1.0 * ANGSTROM),
            ]
        ),
    ],
)
def test_intermolecular_charge_sw_and_ww(system):
    apply_rest(system, {0}, femto.md.config.REST(scale_nonbonded=True))

    energy = compute_energy(
        system,
        TRIANGLE_COORDS,
        None,
        {REST_CTX_PARAM: BM_B0, REST_CTX_PARAM_SQRT: BM_B0_RT},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = (
        ONE_4PI_EPS0
        * (
            # a - b should be scaled by sqrt(BM_B0)
            BM_B0_RT * 2.0 * 3.0 / 3.0
            # a - c should be scaled by sqrt(BM_B0)
            + BM_B0_RT * 2.0 * 4.0 / 4.0
            # b - c should not be scaled
            + 3.0 * 4.0 / 5.0
        )
        * openmm.unit.kilojoules_per_mole
    )
    assert is_close(energy, expected_energy)


# test FEP compatibility
def test_apply_after_fep_simple():
    coords = (
        numpy.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]]
        )
        * openmm.unit.angstrom
    )

    force = openmm.NonbondedForce()
    force.addParticle(0.1, 0.5, 0.9)
    force.addParticle(0.2, 0.6, 0.10)
    force.addParticle(0.3, 0.7, 0.11)
    force.addParticle(0.4, 0.8, 0.12)

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)

    system.addForce(force)

    apply_fep(system, {0, 1}, None, femto.fe.config.FEP())

    state_0 = femto.md.utils.openmm.evaluate_ctx_parameters(
        {
            femto.fe.fep.LAMBDA_CHARGES_LIGAND_1: 0.0,
            femto.fe.fep.LAMBDA_VDW_LIGAND_1: 0.0,
        },
        system,
    )
    state_05 = femto.md.utils.openmm.evaluate_ctx_parameters(
        {
            femto.fe.fep.LAMBDA_CHARGES_LIGAND_1: 0.5,
            femto.fe.fep.LAMBDA_VDW_LIGAND_1: 0.5,
        },
        system,
    )

    energy_0 = femto.md.utils.openmm.compute_energy(system, coords, None, state_0)
    energy_05 = femto.md.utils.openmm.compute_energy(system, coords, None, state_05)

    apply_rest(system, {0, 3}, femto.md.config.REST())

    state_0 = femto.md.utils.openmm.evaluate_ctx_parameters(
        {**state_0, REST_CTX_PARAM: 1.0}, system
    )
    state_05 = femto.md.utils.openmm.evaluate_ctx_parameters(
        {**state_05, REST_CTX_PARAM: 1.0}, system
    )

    energy_0_after = femto.md.utils.openmm.compute_energy(system, coords, None, state_0)
    assert is_close(energy_0, energy_0_after)

    energy_lambda_05_after = femto.md.utils.openmm.compute_energy(
        system, coords, None, state_05
    )
    assert is_close(energy_05, energy_lambda_05_after)

    force: openmm.NonbondedForce
    custom_force: openmm.CustomNonbondedForce

    force, custom_force = system.getForces()
    assert force.getNumParticles() == 4

    expected_ctx_params = [
        "lambda_charges_lig_1",
        "sqrt<bm_b0>*lambda_charges_lig_1",
        "sqrt<bm_b0>",
        "bm_b0",
    ]
    actual_ctx_params = [
        force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())
    ]
    assert sorted(actual_ctx_params) == sorted(expected_ctx_params)

    expected_offsets = [
        # param, idx, q, sig, eps
        [
            "sqrt<bm_b0>*lambda_charges_lig_1",
            0,
            -0.1,
            0.0,
            0.0,
        ],  # REST + FEP Q (-state A)
        ["lambda_charges_lig_1", 1, -0.2, 0.0, 0.0],  # ONLY FEP Q
        ["sqrt<bm_b0>", 0, 0.1, 0.0, 0.0],  # FEP Q (state A)
        ["sqrt<bm_b0>", 3, 0.4, 0.0, 0.0],  # REST Q
        ["bm_b0", 3, 0.0, 0.0, 0.12],  # REST EPS
    ]

    for i in range(force.getNumParticleParameterOffsets()):
        assert force.getParticleParameterOffset(i) == expected_offsets[i]

    expected_exceptions = [
        # idx_1, idx_2, q, sig, eps
        [0, 1, 0.0, 0.55, 0.0],  # scaled by REST - no FEP intrascaling
    ]
    expected_exception_offsets = [
        # i, parameter, q, sig, eps
        ["sqrt<bm_b0>", 0, 0.02, 0.0, 0.3],  # SQRT as sw interaction
    ]

    assert force.getNumExceptions() == len(expected_exceptions)
    assert force.getNumExceptionParameterOffsets() == len(expected_exception_offsets)

    for i in range(force.getNumExceptions()):
        actual_exception = [
            x
            if not isinstance(x, openmm.unit.Quantity)
            else x.value_in_unit_system(openmm.unit.md_unit_system)
            for x in force.getExceptionParameters(i)
        ]
        assert actual_exception == expected_exceptions[i]

        actual_offset = [
            x
            if not isinstance(x, openmm.unit.Quantity)
            else x.value_in_unit_system(openmm.unit.md_unit_system)
            for x in force.getExceptionParameterOffset(i)
        ]
        assert actual_offset == pytest.approx(expected_exception_offsets[i])

    excepted_particles = [
        # q, sig, eps
        [0.0, 0.5, 0.0],  # q handled by offset, eps by custom force
        [0.2, 0.6, 0.0],  # state A charge should be present due to FEP
        [0.3, 0.7, 0.11],  # fully chemical particle with no rest scale
        [0.0, 0.8, 0.0],  # q and eps handled by offset
    ]

    for i in range(force.getNumParticles()):
        actual_particle = [
            x.value_in_unit_system(openmm.unit.md_unit_system)
            for x in force.getParticleParameters(i)
        ]
        assert actual_particle == excepted_particles[i]

    assert custom_force.getNumParticles() == 4
    assert custom_force.getEnergyFunction().startswith("scale1*scale2*4.0")

    expected_ctx_params = [
        "lambda_vdw_lig_1",
        "sqrt<bm_b0>",  # gets used in the LB mixing rule so we need sqrt.
    ]
    actual_ctx_params = [
        custom_force.getGlobalParameterName(i)
        for i in range(custom_force.getNumGlobalParameters())
    ]
    assert sorted(actual_ctx_params) == sorted(expected_ctx_params)

    expected_params = ["eps", "sig", "is_solute"]
    actual_params = [
        custom_force.getPerParticleParameterName(i)
        for i in range(custom_force.getNumPerParticleParameters())
    ]
    assert actual_params == expected_params

    assert custom_force.getNumExclusions() == 1
    assert custom_force.getExclusionParticles(0) == [0, 1]  # from FEP

    excepted_particles = [
        # eps, sig, is_solute
        (0.9, 0.5, 1),  # is rest 'solute'
        (0.10, 0.6, 0),
        (0.11, 0.7, 0),
        (0.12, 0.8, 1),  # is rest 'solute'
    ]

    for i in range(custom_force.getNumParticles()):
        actual_particle = custom_force.getParticleParameters(i)
        assert actual_particle == excepted_particles[i]

    expected_groups = [[(0, 1), (2, 3)]]
    actual_groups = [
        custom_force.getInteractionGroupParameters(i)
        for i in range(custom_force.getNumInteractionGroups())
    ]
    assert actual_groups == expected_groups


# test torsion scaling
@pytest.mark.parametrize(
    "solute_idxs, expected_energy_scale",
    [({0, 1, 2, 3}, BM_B0), ({}, 1.0), ({0, 1}, BM_B0_RT)],
)
def test_torsion_scaling(solute_idxs, expected_energy_scale):
    periodicity = 3
    phase = 175.0 * openmm.unit.degree
    k = 2.0 * openmm.unit.kilojoules_per_mole

    expected_force_group = 17

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.PeriodicTorsionForce()
    force.addTorsion(0, 1, 2, 3, periodicity, phase, k)
    force.setForceGroup(expected_force_group)
    system.addForce(force)

    apply_rest(system, solute_idxs, femto.md.config.REST(scale_torsions=True))
    assert next(iter(system.getForces())).getForceGroup() == expected_force_group

    # theta will be 90 degrees
    coords = (
        numpy.array(
            [[-1.0, 1.0, 0.0], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 1.0]]
        )
        * openmm.unit.angstrom
    )
    theta = 90.0 * openmm.unit.degree

    energy = compute_energy(
        system,
        coords,
        None,
        {REST_CTX_PARAM: BM_B0, REST_CTX_PARAM_SQRT: BM_B0_RT},
        OpenMMPlatform.REFERENCE,
    )
    expected_energy = (
        expected_energy_scale
        * k
        * (
            1
            + numpy.cos((periodicity * theta - phase).value_in_unit(openmm.unit.radian))
        )
    )
    assert is_close(energy, expected_energy)


# test ATM compatibility
@pytest.mark.parametrize("bm_b0", [1.0, 0.5, 0.0])
def test_atm_scaling(bm_b0):
    """Make sure that the REST parameter correctly scales the inner non-bonded
    interactions
    """
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    force.addParticle(0.0, 1.0, 1.0)
    force.addParticle(0.0, 1.0, 1.0)
    force.setForceGroup(OpenMMForceGroup.NONBONDED)
    system.addForce(force)

    coords = numpy.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]) * openmm.unit.angstrom

    energy_0 = compute_energy(system, coords, None, {}, OpenMMPlatform.REFERENCE)
    apply_rest(system, {0, 1}, femto.md.config.REST(scale_nonbonded=True))

    atm_force = openmm.ATMForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
    atm_force.addParticle(openmm.Vec3(0.0, 0.0, 0.0))
    atm_force.addParticle(openmm.Vec3(0.0, 0.0, 0.0))
    atm_force.setForceGroup(OpenMMForceGroup.ATM)
    atm_force.addForce(copy.deepcopy(system.getForce(0)))
    system.addForce(atm_force)
    system.removeForce(0)

    bm_b0_sqrt = numpy.sqrt(bm_b0)

    energy = compute_energy(
        system,
        coords,
        None,
        {REST_CTX_PARAM: bm_b0, REST_CTX_PARAM_SQRT: bm_b0_sqrt},
        OpenMMPlatform.REFERENCE,
    )
    assert is_close(energy, bm_b0 * energy_0)
