"""Prepare a system for REST sampling."""

import collections
import copy
import typing

import openmm
import openmm.unit

import femto.md.config
import femto.md.utils.openmm

_T = typing.TypeVar("_T", bound=openmm.NonbondedForce | openmm.CustomNonbondedForce)


_SUPPORTED_FORCES = [
    openmm.HarmonicBondForce,
    openmm.HarmonicAngleForce,
    openmm.PeriodicTorsionForce,
    openmm.NonbondedForce,
    openmm.CustomNonbondedForce,
    openmm.MonteCarloBarostat,
    openmm.CMMotionRemover,
]


REST_CTX_PARAM = "bm_b0"
"""The global parameter used to scale interactions by beta_m / beta_0 according to
REST2."""
REST_CTX_PARAM_SQRT = "sqrt<bm_b0>"
"""The global parameter used to scale interactions by sqrt(beta_m / beta_0) according
to REST2."""


def _create_torsion_force(
    force: openmm.PeriodicTorsionForce, solute_idxs: set[int]
) -> openmm.CustomTorsionForce:
    """Create a custom torsion force that applies a REST2 scaling to the torsion barrier
    heights"""
    energy_fn = (
        "scale * (k * (1 + cos(periodicity * theta - phase)));"
        f"scale = is_ss * {REST_CTX_PARAM} + is_sw * {REST_CTX_PARAM_SQRT} + is_ww;"
    )

    custom_force = openmm.CustomTorsionForce(energy_fn.replace(" ", ""))
    custom_force.addGlobalParameter(REST_CTX_PARAM, 1.0)
    custom_force.addGlobalParameter(REST_CTX_PARAM_SQRT, 1.0)

    for parameter in ["is_ss", "is_sw", "is_ww"]:
        custom_force.addPerTorsionParameter(parameter)

    custom_force.addPerTorsionParameter("periodicity")
    custom_force.addPerTorsionParameter("phase")
    custom_force.addPerTorsionParameter("k")

    custom_force.setUsesPeriodicBoundaryConditions(
        force.usesPeriodicBoundaryConditions()
    )
    custom_force.setForceGroup(force.getForceGroup())

    for i in range(force.getNumTorsions()):
        *idxs, periodicity, phase, k = force.getTorsionParameters(i)

        is_ss = 1.0 if all(idx in solute_idxs for idx in idxs) else 0.0
        is_sw = 1.0 if not is_ss and any(idx in solute_idxs for idx in idxs) else 0.0
        is_ww = 1.0 if not is_ss and not is_sw else 0.0

        custom_force.addTorsion(*idxs, [is_ss, is_sw, is_ww, periodicity, phase, k])

    return custom_force


def _scale_nonbonded_offsets(force: openmm.NonbondedForce, offset_idxs: list[int]):
    """Modify an existing set of non-bonded parameter offsets to also be scaled by the
    REST2 scaling factor.

    Args:
        force: The non-bonded force to modify.
        offset_idxs: The indices of the parameter offsets to modify.
    """
    import femto.fe.fep

    if len(offset_idxs) == 0:
        return
    if len(offset_idxs) != 1:
        # the current FEP setup only adds a single offset to scale the charge
        raise NotImplementedError("expected at most one parameter offset per particle")

    ctx_parameter, *offset_params = force.getParticleParameterOffset(offset_idxs[0])

    if (
        ctx_parameter != femto.fe.fep.LAMBDA_CHARGES_LIGAND_1
        and ctx_parameter != femto.fe.fep.LAMBDA_CHARGES_LIGAND_2
    ):
        raise NotImplementedError("only FEP charge offsets are supported")

    ctx_parameter = f"{REST_CTX_PARAM_SQRT}*{ctx_parameter}"

    found_ctx_parameters = {
        force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())
    }

    if ctx_parameter not in found_ctx_parameters:
        force.addGlobalParameter(ctx_parameter, 0.0)

    force.setParticleParameterOffset(offset_idxs[0], ctx_parameter, *offset_params)


def _create_default_nonbonded_force(
    force: openmm.NonbondedForce, solute_idxs: set[int]
) -> openmm.NonbondedForce:
    """Create a non-bonded force that applies a REST2 scaling to the epsilon and charge
    parameters."""

    particle_idx_to_offsets: dict[int, list[int]] = collections.defaultdict(list)

    for i in range(force.getNumParticleParameterOffsets()):
        _, idx, *_ = force.getParticleParameterOffset(i)
        particle_idx_to_offsets[idx].append(i)

    force.addGlobalParameter(REST_CTX_PARAM, 1.0)
    force.addGlobalParameter(REST_CTX_PARAM_SQRT, 1.0)

    for idx in solute_idxs:
        charge, sigma, epsilon = force.getParticleParameters(idx)

        if not femto.md.utils.openmm.is_close(charge, charge * 0.0):
            force.addParticleParameterOffset(REST_CTX_PARAM_SQRT, idx, charge, 0.0, 0.0)
            charge *= 0.0

        if not femto.md.utils.openmm.is_close(epsilon, epsilon * 0.0):
            force.addParticleParameterOffset(REST_CTX_PARAM, idx, 0.0, 0.0, epsilon)
            epsilon *= 0.0

        # make sure to scale parameters also scaled by FEP.
        _scale_nonbonded_offsets(force, particle_idx_to_offsets[idx])

        force.setParticleParameters(idx, charge, sigma, epsilon)

    if force.getNumExceptionParameterOffsets() != 0:
        # exceptions are not currently scaled by FEP.
        raise NotImplementedError("exception parameter offsets are not supported")

    for i in range(force.getNumExceptions()):
        *idxs, charge, sigma, epsilon = force.getExceptionParameters(i)

        if all(idx in solute_idxs for idx in idxs):
            ctx_parameter = REST_CTX_PARAM
        elif any(idx in solute_idxs for idx in idxs):
            ctx_parameter = REST_CTX_PARAM_SQRT
        else:
            continue

        if femto.md.utils.openmm.is_close(
            charge, charge * 0.0
        ) and femto.md.utils.openmm.is_close(epsilon, epsilon * 0.0):
            continue

        force.addExceptionParameterOffset(
            ctx_parameter, i, charge, sigma * 0.0, epsilon
        )
        force.setExceptionParameters(i, *idxs, charge * 0.0, sigma, epsilon * 0.0)

    return force


def _create_custom_nonbonded_force(
    force: openmm.CustomNonbondedForce,
    solute_idxs: set[int],
) -> openmm.CustomNonbondedForce:
    """Create a custom non-bonded force that applies a REST2 scaling to the epsilon and
    charge parameters.

    We assume this force has an energy expression that can be linearly scaled by simply
    prepending a multiplicative scaling factor.
    """

    contains_solute_idxs = any(
        len(solute_idxs.union(idxs)) > 0
        for i in range(force.getNumInteractionGroups())
        for idxs in force.getInteractionGroupParameters(i)
    )

    if not contains_solute_idxs:
        return force

    is_solute_param = "is_solute"

    found_parameter_names = {
        force.getPerParticleParameterName(i)
        for i in range(force.getNumPerParticleParameters())
    }
    assert is_solute_param not in found_parameter_names, "param already present"

    force.addGlobalParameter(REST_CTX_PARAM_SQRT, 1.0)
    force.addPerParticleParameter(is_solute_param)

    energy_fn = force.getEnergyFunction()
    energy_fn = (
        f"scale1 * scale2 * {energy_fn};"
        f"scale1 = 1.0-{is_solute_param}1 + {REST_CTX_PARAM_SQRT} * {is_solute_param}1;"
        f"scale2 = 1.0-{is_solute_param}2 + {REST_CTX_PARAM_SQRT} * {is_solute_param}2;"
    )
    force.setEnergyFunction(energy_fn.replace(" ", "").replace(";;", ";"))

    for idx in range(force.getNumParticles()):
        params = force.getParticleParameters(idx)
        params = [*params, 1.0 if idx in solute_idxs else 0.0]
        force.setParticleParameters(idx, params)

    return force


def _create_nonbonded_force(force: _T, solute_idxs: set[int]) -> _T:
    """Create a non-bonded force that applies a REST2 scaling to the epsilon and charge
    parameters."""
    import femto.fe.fep

    force = copy.deepcopy(force)

    found_ctx_parameters = {
        force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())
    }

    supported_ctx_parameters = [
        femto.fe.fep.LAMBDA_VDW_LIGAND_1,
        femto.fe.fep.LAMBDA_VDW_LIGAND_2,
        femto.fe.fep.LAMBDA_CHARGES_LIGAND_1,
        femto.fe.fep.LAMBDA_CHARGES_LIGAND_2,
        openmm.ATMForce.Lambda1(),
        openmm.ATMForce.Lambda2(),
        openmm.ATMForce.Alpha(),
        openmm.ATMForce.Uh(),
        openmm.ATMForce.W0(),
        openmm.ATMForce.Umax(),
        openmm.ATMForce.Ubcore(),
        openmm.ATMForce.Acore(),
        openmm.ATMForce.Direction(),
    ]

    if len(found_ctx_parameters - set(supported_ctx_parameters)) != 0:
        raise NotImplementedError("force contains unrecognised context parameter.")

    if isinstance(force, openmm.NonbondedForce):
        return _create_default_nonbonded_force(force, solute_idxs)
    elif isinstance(force, openmm.CustomNonbondedForce):
        return _create_custom_nonbonded_force(force, solute_idxs)

    raise NotImplementedError


def apply_rest(
    system: openmm.System, solute_idxs: set[int], config: femto.md.config.REST
):
    """Modifies an OpenMM system *in-place* such that the valence, vdW and electrostatic
    interactions are scaled according to the REST2 scheme.

    Namely, torsion barrier heights are scaled by ``beta_m / beta_0``, electrostatic
    charges are scaled by ``sqrt(beta_m / beta_0)``, and LJ epsilons are scaled by
    ``beta_m / beta_0``.

    Notes:
        * REST should be applied after any alchemical modifications.

    Args:
        system: The system to modify in-place.
        solute_idxs: The indices of the 'solute'. Usually this will be the indices of
            the ligand atoms, but may also include protein atoms or only a subet of
            ligand atoms.
        config: Configuration for REST2.
    """

    forces_by_type = collections.defaultdict(dict)

    for i, force in enumerate(system.getForces()):
        if type(force) not in _SUPPORTED_FORCES:
            raise ValueError(f"Force type {type(force)} is not supported for REST.")

        forces_by_type[type(force)][i] = force

    rest_forces = {}

    for force_type, forces in forces_by_type.items():
        for i, force in forces.items():
            if force_type == openmm.HarmonicBondForce and config.scale_bonds:
                raise NotImplementedError("Scaling of bonds is not yet supported.")
            elif force_type == openmm.HarmonicAngleForce and config.scale_angles:
                raise NotImplementedError("Scaling of angles is not yet supported.")
            elif force_type == openmm.PeriodicTorsionForce and config.scale_torsions:
                force = _create_torsion_force(force, solute_idxs)
            elif (
                force_type in (openmm.NonbondedForce, openmm.CustomNonbondedForce)
                and config.scale_nonbonded
            ):
                force = _create_nonbonded_force(force, solute_idxs)
            else:
                force = copy.deepcopy(force)

            rest_forces[i] = force

    for i in reversed(range(system.getNumForces())):
        system.removeForce(i)
    for i in range(len(rest_forces)):
        system.addForce(rest_forces[i])
