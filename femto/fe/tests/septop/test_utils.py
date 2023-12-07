import numpy
import openmm
import pytest

import femto.fe.fep
import femto.fe.septop
import femto.md.constants
import femto.md.rest
import femto.md.utils.openmm


def test_create_state_dicts():
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.NonbondedForce()
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_1, 1.0)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_2, 1.0)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_CHARGES_LIGAND_1, 1.0)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_CHARGES_LIGAND_2, 1.0)
    force.addGlobalParameter(femto.md.rest.REST_CTX_PARAM, 1.0)
    force.addGlobalParameter(femto.md.rest.REST_CTX_PARAM_SQRT, 1.0)
    force.addGlobalParameter(femto.fe.septop._setup.LAMBDA_BORESCH_LIGAND_1, 1.0)
    force.addGlobalParameter(femto.fe.septop._setup.LAMBDA_BORESCH_LIGAND_2, 1.0)
    force.addParticle(1.0, 1.0, 0.0)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.NONBONDED)
    system.addForce(force)

    states = femto.fe.septop.SepTopStates(
        lambda_vdw_ligand_1=[0.0, 0.1],
        lambda_charges_ligand_1=[0.2, 0.3],
        lambda_vdw_ligand_2=[0.4, 0.5],
        lambda_charges_ligand_2=[0.6, 0.7],
        bm_b0=[0.8, 0.9],
        lambda_boresch_ligand_1=[1.0, 1.1],
        lambda_boresch_ligand_2=[1.2, 1.3],
    )
    state_dicts = femto.fe.septop.create_state_dicts(states, system)

    expected_dicts = [
        {
            "bm_b0": 0.8,
            "sqrt<bm_b0>": pytest.approx(numpy.sqrt(0.8)),
            "lambda_charges_lig_1": 0.2,
            "lambda_charges_lig_2": 0.6,
            "lambda_vdw_lig_1": 0.0,
            "lambda_vdw_lig_2": 0.4,
            "lambda_boresch_lig_1": 1.0,
            "lambda_boresch_lig_2": 1.2,
        },
        {
            "bm_b0": 0.9,
            "sqrt<bm_b0>": pytest.approx(numpy.sqrt(0.9)),
            "lambda_charges_lig_1": 0.3,
            "lambda_charges_lig_2": 0.7,
            "lambda_vdw_lig_1": 0.1,
            "lambda_vdw_lig_2": 0.5,
            "lambda_boresch_lig_1": 1.1,
            "lambda_boresch_lig_2": 1.3,
        },
    ]
    assert state_dicts == expected_dicts
