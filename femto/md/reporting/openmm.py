"""Reporters compatible with OpenMM simulation objects"""
import numpy
import openmm.app
import openmm.unit

from femto.md.reporting import Reporter


class OpenMMStateReporter:
    """Report OpenMM simulation state using a Reporter object."""

    def __init__(
        self,
        reporter: Reporter,
        tag: str,
        interval: int,
        total_energy: bool = True,
        potential_energy: bool = True,
        kinetic_energy: bool = False,
        volume: bool = True,
        temperature: bool = True,
    ):
        """

        Args:
            reporter: The reporter to use.
            tag: The tag to report using.
            interval: The interval at which to report.
            total_energy: Report the total energy.
            potential_energy: Report the potential energy.
            kinetic_energy: Report the kinetic energy.
            volume: Report the volume.
            temperature: Report the temperature.
        """
        self._has_initialized = False
        self._dof = None

        self._reporter = reporter

        self.tag = tag
        self.interval = interval

        self._include_total_energy = total_energy
        self._include_potential_energy = potential_energy
        self._include_kinetic_energy = kinetic_energy
        self._include_volume = volume
        self._include_temperature = temperature

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return steps, False, False, False, True

    def _initialize_constants(self, simulation):
        system = simulation.system

        dof = sum(
            3
            for i in range(system.getNumParticles())
            if system.getParticleMass(i) > 0 * openmm.unit.dalton
        )

        for i in range(system.getNumConstraints()):
            p1, p2, _ = system.getConstraintParameters(i)

            if (
                system.getParticleMass(p1) > 0 * openmm.unit.dalton
                or system.getParticleMass(p2) > 0 * openmm.unit.dalton
            ):
                dof -= 1

        if any(
            type(system.getForce(i)) == openmm.CMMotionRemover
            for i in range(system.getNumForces())
        ):
            dof -= 3

        self._dof = dof

    def report(self, simulation, state):
        if not self._has_initialized:
            self._initialize_constants(simulation)
            self._has_initialized = True

        e_potential = state.getPotentialEnergy()
        e_kinetic = state.getKineticEnergy()

        total_energy = (e_potential + e_kinetic).value_in_unit(
            openmm.unit.kilocalorie_per_mole
        )

        if numpy.isnan(total_energy):
            raise ValueError("Energy is NaN.")
        if numpy.isinf(total_energy):
            raise ValueError("Energy is infinite.")

        step = simulation.currentStep

        box = state.getPeriodicBoxVectors()
        volume = box[0][0] * box[1][1] * box[2][2]

        if self._include_total_energy:
            self._reporter.report_scalar(
                f"{self.tag}/E_tot_kcal_mol", step, total_energy
            )
        if self._include_potential_energy:
            self._reporter.report_scalar(
                f"{self.tag}/E_pot_kcal_mol",
                step,
                e_potential.value_in_unit(openmm.unit.kilocalorie_per_mole),
            )
        if self._include_kinetic_energy:
            self._reporter.report_scalar(
                f"{self.tag}/E_kin_kcal_mol",
                step,
                e_kinetic.value_in_unit(openmm.unit.kilocalorie_per_mole),
            )
        if self._include_volume:
            self._reporter.report_scalar(
                f"{self.tag}/volume_nm3",
                step,
                volume.value_in_unit(openmm.unit.nanometer**3),
            )
        if self._include_temperature:
            integrator = simulation.context.getIntegrator()

            temperature = (
                integrator.computeSystemTemperature()
                if hasattr(integrator, "computeSystemTemperature")
                else (
                    2
                    * state.getKineticEnergy()
                    / (self._dof * openmm.unit.MOLAR_GAS_CONSTANT_R)
                )
            )

            self._reporter.report_scalar(
                f"{self.tag}/T_K", step, temperature.value_in_unit(openmm.unit.kelvin)
            )
