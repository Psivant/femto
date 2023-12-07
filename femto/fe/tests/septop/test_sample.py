import numpy
import openmm
import parmed
import pytest
from pymbar.testsystems import harmonic_oscillators

import femto.fe.config
import femto.fe.fep
import femto.fe.septop
import femto.md.constants
import femto.md.reporting
import femto.md.rest
import femto.md.utils.openmm
from femto.fe.septop._sample import run_hremd, _analyze
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture()
def mock_system() -> openmm.System:
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_1, 1.0)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_2, 1.0)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_CHARGES_LIGAND_1, 1.0)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_CHARGES_LIGAND_2, 1.0)
    force.addGlobalParameter(femto.md.rest.REST_CTX_PARAM, 1.0)
    force.addGlobalParameter(femto.md.rest.REST_CTX_PARAM_SQRT, 1.0)
    force.addGlobalParameter(femto.fe.septop.LAMBDA_BORESCH_LIGAND_1, 1.0)
    force.addGlobalParameter(femto.fe.septop.LAMBDA_BORESCH_LIGAND_2, 1.0)
    force.addParticle(0.0, 1.0, 0.0)
    system = openmm.System()
    system.addParticle(1.0)
    system.addForce(force)
    return system


@pytest.fixture()
def mock_topology(mock_system) -> parmed.Structure:
    topology = build_mock_structure(["[Ar]"])
    topology.coordinates = [[0.0, 0.0, 0.0]]
    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME
    topology.box_vectors = mock_system.getDefaultPeriodicBoxVectors()

    return topology


@pytest.fixture()
def mock_coords(mock_system, mock_topology):
    context = openmm.Context(mock_system, openmm.VerletIntegrator(0.001))
    context.setPeriodicBoxVectors(*mock_topology.box_vectors)
    context.setPositions(mock_topology.positions)

    state = context.getState(getPositions=True)

    return state


def test_analyze(mocker):
    n_states = 22

    testcase = harmonic_oscillators.HarmonicOscillatorsTestCase(
        [0.0] * n_states, [1.0] * n_states
    )
    _, u_kn, n_k, _ = testcase.sample([10] * n_states, mode="u_kn")

    config = femto.fe.septop.SepTopConfig().solution.sample
    reporter = mocker.MagicMock(spec=femto.md.reporting.Reporter)

    cycle = 1243
    _analyze(cycle, u_kn, n_k, config, reporter)

    reporter.report_scalar.assert_has_calls(
        [
            mocker.call("ddG_kcal_mol", cycle, pytest.approx(0.0, abs=1.0e-7)),
            mocker.call("ddG_error_kcal_mol", cycle, pytest.approx(0.0, abs=1.0e-7)),
        ]
    )


def test_run_hremd(tmp_cwd, mocker, mock_system, mock_topology, mock_coords):
    mock_run_hremd = mocker.patch("femto.md.hremd.run_hremd", autospec=True)

    config = femto.fe.septop.SepTopSamplingStage()
    states = femto.fe.septop.SepTopStates(
        lambda_vdw_ligand_1=[0.0, 0.1],
        lambda_charges_ligand_1=[0.2, 0.3],
        lambda_vdw_ligand_2=[0.4, 0.5],
        lambda_charges_ligand_2=[0.6, 0.7],
        bm_b0=[0.8, 0.9],
        lambda_boresch_ligand_1=[1.0, 1.1],
        lambda_boresch_ligand_2=[1.2, 1.3],
    )

    expected_output_dir = tmp_cwd / "outputs"

    run_hremd(
        mock_system,
        mock_topology,
        [mock_coords],
        states,
        config,
        "CPU",
        expected_output_dir,
    )

    mock_run_hremd.assert_called_once()

    simulation, states, config, output_dir = mock_run_hremd.call_args.args

    expected_states = [
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
    assert states == expected_states

    assert output_dir == expected_output_dir
