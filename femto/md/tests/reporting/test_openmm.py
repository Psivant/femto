import numpy
import openmm
import openmm.unit
import pytest

from femto.md.reporting import NullReporter
from femto.md.reporting.openmm import OpenMMStateReporter


class TestOpenMMStateReporter:
    @pytest.fixture
    def mock_reporter(self) -> OpenMMStateReporter:
        return OpenMMStateReporter(
            NullReporter(),
            "mock/tag",
            interval=5,
            total_energy=True,
            potential_energy=True,
            kinetic_energy=True,
            volume=True,
            temperature=True,
        )

    def test_describe_next_report(self, mock_reporter, mocker):
        mock_simulation = mocker.MagicMock()
        mock_simulation.currentStep = 2

        (
            steps,
            get_positions,
            get_velocities,
            get_forces,
            get_energy,
        ) = mock_reporter.describeNextReport(mock_simulation)

        assert steps == 3  # interval - step

        assert get_positions is False
        assert get_velocities is False
        assert get_forces is False
        assert get_energy is True  # only report energy based statistics.

    def test_initialize_constants(self, mock_reporter, mocker):
        mock_system = openmm.System()
        mock_system.addParticle(0.0)
        mock_system.addParticle(12.0)
        mock_system.addParticle(0.0)
        mock_system.addParticle(12.0)
        mock_system.addForce(openmm.CMMotionRemover())

        mock_system.addConstraint(0, 2, 0.1)
        mock_system.addConstraint(1, 3, 0.2)

        mock_simulation = mocker.MagicMock()
        mock_simulation.system = mock_system

        mock_reporter._initialize_constants(mock_simulation)

        # particles - constraints - com force
        expected_dof = 0 + 3 + 0 + 3 - 0 - 1 - 3
        assert mock_reporter._dof == expected_dof

    def test_report(self, mock_reporter, mocker):
        expected_step = 2

        mock_system = openmm.System()
        mock_system.addParticle(1.0)

        mock_simulation = mocker.MagicMock()
        mock_simulation.system = mock_system
        mock_simulation.currentStep = expected_step

        mock_state = mocker.MagicMock()

        expected_potential = 1.0 * openmm.unit.kilojoules_per_mole
        mock_state.getPotentialEnergy.return_value = expected_potential
        expected_kinetic = 2.0 * openmm.unit.kilojoules_per_mole
        mock_state.getKineticEnergy.return_value = expected_kinetic

        expected_box = (numpy.eye(3) * 3.0) * openmm.unit.angstrom
        expected_volume = (3.0**3) * (openmm.unit.angstrom**3)

        expected_temperature = 150.0 * openmm.unit.kelvin

        mock_integrator = (
            mock_simulation.context.getIntegrator.return_value.computeSystemTemperature
        )
        mock_integrator.return_value = expected_temperature

        mock_state.getPeriodicBoxVectors.return_value = expected_box

        mock_reporter._reporter = mocker.MagicMock()
        mock_reporter.report(mock_simulation, mock_state)

        expected_tag = mock_reporter.tag

        report_scalar_calls = {
            call_args.args[0]: call_args.args[1:]
            for call_args in mock_reporter._reporter.report_scalar.call_args_list
        }

        expected_values = {
            f"{expected_tag}/E_pot_kcal_mol": expected_potential,
            f"{expected_tag}/E_kin_kcal_mol": expected_kinetic,
            f"{expected_tag}/E_tot_kcal_mol": expected_potential + expected_kinetic,
            f"{expected_tag}/volume_nm3": expected_volume,
            f"{expected_tag}/T_K": expected_temperature,
        }
        expected_units = {
            f"{expected_tag}/E_pot_kcal_mol": openmm.unit.kilocalorie_per_mole,
            f"{expected_tag}/E_kin_kcal_mol": openmm.unit.kilocalorie_per_mole,
            f"{expected_tag}/E_tot_kcal_mol": openmm.unit.kilocalorie_per_mole,
            f"{expected_tag}/volume_nm3": openmm.unit.nanometer**3,
            f"{expected_tag}/T_K": openmm.unit.kelvin,
        }

        assert {*report_scalar_calls} == {*expected_values}

        for tag in expected_values:
            step, value = report_scalar_calls[tag]
            assert step == expected_step

            expected_value = expected_values[tag].value_in_unit(expected_units[tag])
            assert numpy.isclose(value, expected_value)

    @pytest.mark.parametrize(
        "energy, match_str", [(numpy.nan, "NaN"), (numpy.inf, "infinite")]
    )
    def test_report_nan(self, energy, match_str, mock_reporter, mocker):
        mock_reporter._has_initialized = True

        mock_simulation = mocker.MagicMock()
        mock_state = mocker.MagicMock()

        expected_potential = energy * openmm.unit.kilojoules_per_mole
        mock_state.getPotentialEnergy.return_value = expected_potential
        expected_kinetic = 2.0 * openmm.unit.kilojoules_per_mole
        mock_state.getKineticEnergy.return_value = expected_kinetic

        with pytest.raises(ValueError, match=f"Energy is {match_str}."):
            mock_reporter.report(mock_simulation, mock_state)
