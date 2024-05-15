import json
import warnings
from pathlib import Path
from typing import Union, Optional, Sequence, Callable, Dict, Any

import numpy as np
import pandas as pd

from first_breaks.picking.picks import Picks
from first_breaks.sgy.headers import TraceHeaders
from first_breaks.sgy.reader import SGY


def export_to_sgy(
    sgy: SGY,
    filename: Union[Path, str],
    picks: Picks,
    byte_position: int = 236,
    encoding: Optional[str] = None,
    picks_unit: Optional[str] = "mcs",
) -> None:
    sgy.export_sgy_with_picks(
        output_fname=filename,
        picks_in_mcs=picks.picks_in_mcs,
        byte_position=byte_position,
        encoding=encoding,
        picks_unit=picks_unit,
    )


COL_PICKS_IN_MCS = "Picks, mcs"
COL_PICKS_IN_MS = "Picks, ms"
COL_PICKS_IN_SAMPLES = "Picks, samples"
COL_PICKS_CONFIDENCE = "Picks, confidence"
PICKS_COLUMNS = [
    COL_PICKS_IN_MCS,
    COL_PICKS_IN_MS,
    COL_PICKS_IN_SAMPLES,
    COL_PICKS_CONFIDENCE,
]


def _prepare_column_values_to_export(
    picks: Picks,
    columns: Sequence[str] = (COL_PICKS_IN_MCS,),
    sgy: Optional[SGY] = None,
    process_values: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    if not any(v in PICKS_COLUMNS for v in columns):
        warnings.warn(
            f"No pick columns were selected. Add them using their names {PICKS_COLUMNS}, " f"or ignore the warning."
        )

    available_columns = PICKS_COLUMNS.copy()
    traces_columns = [name for pos, name, encoding in TraceHeaders().headers_schema]
    available_columns = available_columns + traces_columns
    unsupported_columns = set(columns) - set(available_columns)
    if unsupported_columns:
        raise ValueError(f"Unsupported columns: {unsupported_columns}. Only {available_columns} are available")

    if any(v in traces_columns for v in columns) and sgy is None:
        raise ValueError("Columns from the file were selected. You need to pass the SGY.")

    to_export = {}

    for column in columns:
        if column in PICKS_COLUMNS:
            if column == COL_PICKS_IN_MCS:
                value = picks.picks_in_mcs
            elif column == COL_PICKS_IN_MS:
                value = picks.picks_in_ms
            elif column == COL_PICKS_IN_SAMPLES:
                value = picks.picks_in_samples
            elif column == COL_PICKS_CONFIDENCE:
                value = picks.confidence if picks.confidence is not None else [None] * len(picks)
            else:
                raise ValueError("Unsupported column")
        else:
            value = sgy.traces_headers[column]

        value = np.array(value)

        if process_values:
            value = process_values(value)

        if column in [COL_PICKS_IN_MCS, COL_PICKS_IN_SAMPLES]:
            value = value.astype(int)
        value = value.tolist()

        to_export[column] = value

    return to_export


def export_to_txt(
    filename: Union[Path, str],
    picks: Picks,
    columns: Sequence[str] = (COL_PICKS_IN_MCS,),
    sgy: Optional[SGY] = None,
    separator: str = "\t",
    include_column_names: bool = True,
    precision: int = 3,
):
    to_export = _prepare_column_values_to_export(
        picks=picks,
        columns=columns,
        sgy=sgy,
        process_values=lambda x: np.round(x, precision),
    )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(to_export).to_csv(filename, index=False, sep=separator, header=include_column_names)


def export_to_json(
    filename: Union[Path, str],
    picks: Picks,
    columns: Sequence[str] = (COL_PICKS_IN_MCS,),
    sgy: Optional[SGY] = None,
    include_picking_parameters: bool = True,
    extra_data: Optional[Dict[str, Any]] = None,
):
    to_export = _prepare_column_values_to_export(
        picks=picks,
        columns=columns,
        sgy=sgy,
        process_values=None,
    )
    if include_picking_parameters and picks.picking_parameters is not None:
        to_export["picking_parameters"] = picks.picking_parameters.model_dump()

    if extra_data:
        to_export.update(extra_data)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(to_export, f)
