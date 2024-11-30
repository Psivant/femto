import numpy
import openmm
import openmm.app
import openmm.unit
import pytest

import femto.fe.config
import femto.fe.fep
import femto.fe.septop
import femto.md.anneal
import femto.md.config
import femto.md.constants
import femto.md.reporting
import femto.md.reporting.openmm
import femto.md.utils.openmm
import femto.top
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture
def mock_topology() -> femto.top.Topology:
    topology = build_mock_structure(["[Ar]"])
    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME
    topology.box = numpy.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])

    return topology


@pytest.fixture
def mock_system(mock_topology) -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0)

    system.setDefaultPeriodicBoxVectors(*mock_topology.box)

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_1, 1.0)
    force.addParticle(1.0, 1.0, 0.0)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.NONBONDED)
    system.addForce(force)

    return system


def test_equilibrate_states(mock_system, mock_topology, mocker):
    n_expected_states = 2

    mock_coords = [mocker.Mock()] * n_expected_states
    mock_simulate = mocker.patch(
        "femto.md.simulate.simulate_states",
        return_value=mock_coords,
    )
    mock_stages = [femto.md.config.Minimization()]

    mock_config = femto.fe.septop.SepTopEquilibrateStage(stages=mock_stages)
    mock_states = femto.fe.septop.SepTopStates(
        lambda_vdw_ligand_1=[0.0] * n_expected_states,
        lambda_charges_ligand_1=[0.0] * n_expected_states,
        lambda_vdw_ligand_2=[0.0] * n_expected_states,
        lambda_charges_ligand_2=[0.0] * n_expected_states,
        lambda_boresch_ligand_1=None,
        lambda_boresch_ligand_2=None,
    )

    outputs = femto.fe.septop.equilibrate_states(
        mock_system,
        mock_topology,
        mock_states,
        mock_config,
        platform="Reference",
    )
    mock_simulate.assert_called_once_with(
        mock_system, mock_topology, mocker.ANY, mock_stages, "Reference", mocker.ANY
    )

    assert outputs == mock_coords
