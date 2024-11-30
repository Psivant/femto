import pathlib
import shutil

import femto.fe.inputs
import femto.md.utils.models

_DATA_DIR = pathlib.Path(__file__).parent / "data"

TEMOA_DATA_DIR: pathlib.Path = (_DATA_DIR / "temoa").resolve()
CDK2_DATA_DIR: pathlib.Path = (_DATA_DIR / "cdk2").resolve()


class TestSystem(femto.md.utils.models.BaseModel):
    directory: pathlib.Path
    """The directory containing the test system data."""

    receptor_name: str
    """The name of the receptor."""
    receptor_coords: pathlib.Path
    """The path to the receptor coordinates."""
    receptor_params: pathlib.Path | None
    """The path to the receptor parameters."""
    receptor_cavity_mask: str
    """An AMBER style query mask that selects the receptor atoms that form the binding
    site."""
    receptor_ref_atoms: tuple[str, str, str]
    """The AMBER style query masks that select the receptors' reference atoms that can
    be used with Boresch style alignment restraints."""

    ligand_1_name: str
    """The name of the first ligand."""
    ligand_1_coords: pathlib.Path
    """The path to the first ligand coordinates."""
    ligand_1_params: pathlib.Path
    """The path to the first ligand parameters."""
    ligand_1_ref_atoms: tuple[str, str, str]
    """The AMBER style query masks that select the first ligands' reference atoms."""

    ligand_2_name: str
    """The name of the second ligand."""
    ligand_2_coords: pathlib.Path
    """The path to the second ligand coordinates."""
    ligand_2_params: pathlib.Path
    """The path to the second ligand parameters."""
    ligand_2_ref_atoms: tuple[str, str, str]
    """The AMBER style query masks that select the second ligands' reference atoms."""

    @property
    def _receptor_inputs(self) -> femto.fe.inputs.Structure:
        return femto.fe.inputs.Structure(
            name=self.receptor_name,
            coords=self.receptor_coords,
            params=self.receptor_params,
            metadata={"ref_atoms": self.receptor_ref_atoms},
        )

    @property
    def _ligand_1_inputs(self) -> femto.fe.inputs.Structure:
        return femto.fe.inputs.Structure(
            name=self.ligand_1_name,
            coords=self.ligand_1_coords,
            params=self.ligand_1_params,
            metadata={"ref_atoms": self.ligand_1_ref_atoms},
        )

    @property
    def _ligand_2_inputs(self) -> femto.fe.inputs.Structure:
        return femto.fe.inputs.Structure(
            name=self.ligand_2_name,
            coords=self.ligand_2_coords,
            params=self.ligand_2_params,
            metadata={"ref_atoms": self.ligand_2_ref_atoms},
        )

    @property
    def abfe_network(self) -> femto.fe.inputs.Network:
        """Returns an object representing the ``ligand_1 -> None`` edge for this
        system."""

        return femto.fe.inputs.Network(
            receptor=self._receptor_inputs,
            edges=[femto.fe.inputs.Edge(ligand_1=self._ligand_1_inputs, ligand_2=None)],
        )

    @property
    def rbfe_network(self) -> femto.fe.inputs.Network:
        """Returns an object representing the ``ligand_1 -> None`` edge for this
        system."""

        return femto.fe.inputs.Network(
            receptor=self._receptor_inputs,
            edges=[
                femto.fe.inputs.Edge(
                    ligand_1=self._ligand_1_inputs, ligand_2=self._ligand_2_inputs
                )
            ],
        )


TEMOA_SYSTEM = TestSystem(
    directory=TEMOA_DATA_DIR,
    receptor_name="temoa",
    receptor_coords=TEMOA_DATA_DIR / "temoa.sdf",
    receptor_params=TEMOA_DATA_DIR / "temoa.parm7",
    receptor_cavity_mask="@1-40",
    receptor_ref_atoms=("@1", "@2", "@3"),
    ligand_1_name="g1",
    ligand_1_coords=TEMOA_DATA_DIR / "g1.mol2",
    ligand_1_params=TEMOA_DATA_DIR / "g1.parm7",
    ligand_1_ref_atoms=("@8", "@6", "@4"),
    ligand_2_name="g4",
    ligand_2_coords=TEMOA_DATA_DIR / "g4.sdf",
    ligand_2_params=TEMOA_DATA_DIR / "g4.parm7",
    ligand_2_ref_atoms=("@3", "@5", "@1"),
)
CDK2_SYSTEM = TestSystem(
    directory=CDK2_DATA_DIR,
    receptor_name="cdk2",
    receptor_coords=CDK2_DATA_DIR / "cdk2.pdb",
    receptor_params=None,
    receptor_cavity_mask=":12,14,16,22,84,87,88,134,146,147 & @CA",
    receptor_ref_atoms=("@1", "@2", "@3"),
    ligand_1_name="1h1q",
    ligand_1_coords=CDK2_DATA_DIR / "1h1q.sdf",
    ligand_1_params=CDK2_DATA_DIR / "1h1q.parm7",
    ligand_1_ref_atoms=("@14", "@21", "@18"),
    ligand_2_name="1oiu",
    ligand_2_coords=CDK2_DATA_DIR / "1oiu.sdf",
    ligand_2_params=CDK2_DATA_DIR / "1oiu.parm7",
    ligand_2_ref_atoms=("@16", "@23", "@20"),
)


def _create_standard_inputs(
    root_dir: pathlib.Path, system: TestSystem, ligand_coord_suffix: str
):
    """Create a standard BFE directory structure with the given inputs.

    Notes:
        Neither a ``edges.yaml`` nor a ``Morph.in`` file will be created.
    """

    root_dir.mkdir(exist_ok=True, parents=True)

    output_receptor_path = (
        root_dir
        / "proteins"
        / system.receptor_name
        / f"protein{system.receptor_coords.suffix}"
    )
    output_receptor_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(system.receptor_coords, output_receptor_path)

    if system.receptor_params is not None:
        shutil.copyfile(
            system.receptor_params, output_receptor_path.with_suffix(".parm7")
        )

    raise NotImplementedError

    # ligands = [
    #     (system.ligand_1_name, system.ligand_1_coords, system.ligand_1_params),
    #     (system.ligand_2_name, system.ligand_2_coords, system.ligand_2_params),
    # ]
    #
    # for ligand_name, ligand_coords, ligand_params in ligands:
    #     ligand_dir = root_dir / "forcefield" / ligand_name
    #     ligand_dir.mkdir(exist_ok=True, parents=True)
    #
    #     structure = parmed.amber.AmberParm(str(ligand_params), str(ligand_coords))
    #     structure.save(str(ligand_dir / "vacuum.parm7"), overwrite=True)
    #     structure.save(
    #         str(ligand_dir / f"vacuum.{ligand_coord_suffix}"), overwrite=True
    #     )
    #
    # return root_dir


def create_temoa_input_directory(
    root_dir: pathlib.Path, ligand_coord_suffix: str = "rst7"
):
    """Create a directory structure containing the TEMOA input files"""
    _create_standard_inputs(root_dir, TEMOA_SYSTEM, ligand_coord_suffix)


def create_cdk2_input_directory(
    root_dir: pathlib.Path, ligand_coord_suffix: str = "mol2"
):
    """Create a directory structure containing the CDK2 input files"""
    _create_standard_inputs(root_dir, CDK2_SYSTEM, ligand_coord_suffix)
