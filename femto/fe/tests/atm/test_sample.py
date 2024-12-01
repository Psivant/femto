import numpy
import openmm
import pytest
from pymbar.testsystems import harmonic_oscillators

import femto.fe.config
import femto.md.config
import femto.md.constants
import femto.md.reporting
import femto.md.rest
import femto.md.utils.openmm
import femto.top
from femto.fe.atm._sample import _analyze, run_hremd
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture()
def mock_system() -> openmm.System:
    force = openmm.NonbondedForce()
    force.addParticle(0.0, 1.0, 0.0)
    system = openmm.System()
    system.addParticle(1.0)
    system.addForce(force)
    return system


@pytest.fixture()
def mock_topology(mock_system) -> femto.top.Topology:
    topology = build_mock_structure(["[Ar]"])
    topology.xyz = numpy.array([[0.0, 0.0, 0.0]]) * openmm.unit.angstrom
    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME

    box = [
        v.value_in_unit(openmm.unit.angstrom)
        for v in mock_system.getDefaultPeriodicBoxVectors()
    ]
    topology.box = box

    return topology


@pytest.fixture()
def mock_coords(mock_system, mock_topology):
    context = openmm.Context(mock_system, openmm.VerletIntegrator(0.001))
    context.setPeriodicBoxVectors(*mock_topology.box)
    context.setPositions(mock_topology.xyz)

    state = context.getState(getPositions=True)

    return [state, state]


def test_analyze(mocker):
    n_states = 22

    testcase = harmonic_oscillators.HarmonicOscillatorsTestCase(
        [0.0] * n_states, [1.0] * n_states
    )
    _, u_kn, n_k, _ = testcase.sample([10] * n_states, mode="u_kn")

    config = femto.fe.atm.ATMConfig()
    reporter = mocker.MagicMock(spec=femto.md.reporting.Reporter)

    cycle = 1243
    _analyze(cycle, u_kn, n_k, config.sample, config.states, reporter)

    reporter.report_scalar.assert_has_calls(
        [
            mocker.call("ddG_kcal_mol", cycle, pytest.approx(0.0, abs=1.0e-7)),
            mocker.call("ddG_error_kcal_mol", cycle, pytest.approx(0.0, abs=1.0e-7)),
        ]
    )


def test_run_hremd(tmp_cwd, mocker, mock_system, mock_topology, mock_coords):
    mock_run_hremd = mocker.patch("femto.md.hremd.run_hremd", autospec=True)

    states = femto.fe.atm.ATMStates(
        lambda_1=[0.0, 0.1],
        lambda_2=[0.2, 0.3],
        direction=[1, -1],
        alpha=[0.01, 0.02],
        u0=[0.5, 0.6],
        w0=[2.0, 2.1],
        bm_b0=[0.5, 0.6],
    )

    expected_output_dir = tmp_cwd / "outputs"

    femto.md.rest.apply_rest(mock_system, {0}, femto.md.config.REST())

    run_hremd(
        mock_system,
        mock_topology,
        mock_coords,
        states,
        femto.fe.atm.ATMSamplingStage(),
        [0.0, 0.0, 0.0] * openmm.unit.angstrom,
        "CPU",
        expected_output_dir,
    )

    mock_run_hremd.assert_called_once()

    (
        simulation,
        states,
        config,
        output_dir,
        swap_mask,
    ) = mock_run_hremd.call_args.args

    hremd_system = simulation.context.getSystem()
    assert femto.md.constants.OpenMMForceGroup.ATM in {
        force.getForceGroup() for force in hremd_system.getForces()
    }

    expected_states = [
        {
            "Alpha": 0.002390057361376673,
            "Direction": 1,
            "Lambda1": 0.0,
            "Lambda2": 0.2,
            "Uh": 2.092,
            "W0": 8.368,
            "bm_b0": 0.5,
        },
        {
            "Alpha": 0.004780114722753346,
            "Direction": -1,
            "Lambda1": 0.1,
            "Lambda2": 0.3,
            "Uh": 2.5104,
            "W0": 8.7864,
            "bm_b0": 0.6,
        },
    ]
    assert states == expected_states
    assert swap_mask == {(0, 1), (1, 0)}

    assert output_dir == expected_output_dir
