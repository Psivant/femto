"""Utilities for preparing SepTop simulations."""

import typing

import openmm.unit

import femto.fe.config
import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.reporting
import femto.md.rest
import femto.md.utils.openmm
from femto.fe.fep import (
    LAMBDA_CHARGES_LIGAND_1,
    LAMBDA_CHARGES_LIGAND_2,
    LAMBDA_VDW_LIGAND_1,
    LAMBDA_VDW_LIGAND_2,
)

if typing.TYPE_CHECKING:
    import femto.fe.septop


def create_state_dicts(
    config: "femto.fe.septop.SepTopStates", system: openmm.System
) -> list[dict[str, float]]:
    """Map the lambda states specified in the configuration to a dictionary.

    Args:
        config: The configuration.
        system: The system being simulated.

    Returns:
        The dictionary of lambda states.
    """
    from femto.fe.septop import LAMBDA_BORESCH_LIGAND_1, LAMBDA_BORESCH_LIGAND_2

    states = [
        {
            LAMBDA_VDW_LIGAND_1: config.lambda_vdw_ligand_1[i],
            LAMBDA_CHARGES_LIGAND_1: config.lambda_charges_ligand_1[i],
            **(
                {LAMBDA_VDW_LIGAND_2: config.lambda_vdw_ligand_2[i]}
                if config.lambda_vdw_ligand_2 is not None
                else {}
            ),
            **(
                {LAMBDA_CHARGES_LIGAND_2: config.lambda_charges_ligand_2[i]}
                if config.lambda_charges_ligand_2 is not None
                else {}
            ),
            **(
                {LAMBDA_BORESCH_LIGAND_1: config.lambda_boresch_ligand_1[i]}
                if config.lambda_boresch_ligand_1 is not None
                else {}
            ),
            **(
                {LAMBDA_BORESCH_LIGAND_2: config.lambda_boresch_ligand_2[i]}
                if config.lambda_boresch_ligand_2 is not None
                else {}
            ),
            **(
                {femto.md.rest.REST_CTX_PARAM: config.bm_b0[i]}
                if config.bm_b0 is not None
                else {}
            ),
        }
        for i in range(len(config.lambda_vdw_ligand_1))
    ]

    return [
        femto.md.utils.openmm.evaluate_ctx_parameters(state, system) for state in states
    ]
