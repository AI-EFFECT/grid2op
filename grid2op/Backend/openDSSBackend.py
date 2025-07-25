# This file is part of German-node (opendss-backend) and is
# used to provide a mapping between opendss and grid2op.

import os  # load the python os default module

from matplotlib.pyplot import disconnect
import numpy as np
from typing import Optional, Union, Tuple

import py_dss_interface as DSS
import grid2op
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend.backend import Backend

from grid2op.Backend.openDSSDFs import Elements, Powers, Voltages, Currents

ERR_MSG_ELSEWHERE = "Will be detailed in another example script"


class OpenDSSBackend(Backend):

    shunts_data_available = False

    def load_grid(self,
                  path: Union[os.PathLike, str],
                  filename: Optional[Union[os.PathLike, str]] = None) -> None:

        # TODO Added these avoid warnings during self.assert_grid_correct()
        # but not sure what they are supposed to actually do!
        self.can_handle_more_than_2_busbar()
        self.can_handle_detachment()

        # Initialize OpenDSS interface
        full_path = path
        if filename is not None:
            full_path = os.path.join(full_path, filename)
        self._grid = DSS.DSS()
        self._grid.text(f"Compile {full_path}")

        # Get element dataframes
        el_dfs = Elements(self._grid)
        # TODO make these local dfs and not self attributes
        self._loads_df, self._gens_df, self._lines_df, self._trafos_df = \
            el_dfs.loads_df, el_dfs.generators_df, el_dfs.lines_df, el_dfs.transformers_df

        # Get names of all grid elements
        # - each bus becomes a substation in Grid2Op
        self.name_sub = np.array(self._grid.circuit.buses_names, dtype=str)
        self.name_gen = np.array(self._gens_df.name, dtype=str)
        self.name_load = np.array(self._loads_df.name, dtype=str)
        # - grid2op considers trafos as powerlines
        self.name_line = np.concatenate(
            (self._lines_df.name.map(str.lower), self._trafos_df.name.map(str.lower))
        )

        # Get count of all grid elements
        self.n_sub, self.n_gen, self.n_load, self.n_line = \
            len(self.name_sub), len(self.name_gen), len(self.name_load), len(self.name_line)

        # Get location of loads, generators and lines
        sub_name2id = dict(zip(list(map(str.lower, self.name_sub)), range(self.n_sub)))
        self.load_to_subid = np.array(self._loads_df.bus1.map(str.lower).map(sub_name2id), dtype=dt_int)
        self.gen_to_subid = np.array(self._gens_df.bus1.map(str.lower).map(sub_name2id), dtype=dt_int)
        self.line_or_to_subid = np.array(self._lines_df.bus1.map(str.lower).map(sub_name2id), dtype=dt_int)
        self.line_ex_to_subid = np.array(self._lines_df.bus2.map(str.lower).map(sub_name2id), dtype=dt_int)
        # - grid2op considers trafos as powerlines
        trafo_or_to_subid = np.array(self._trafos_df.bus1.map(str.lower).map(sub_name2id), dtype=dt_int)
        trafo_ex_to_subid = np.array(self._trafos_df.bus2.map(str.lower).map(sub_name2id), dtype=dt_int)
        self.line_or_to_subid = np.concatenate((self.line_or_to_subid, trafo_or_to_subid))
        self.line_ex_to_subid = np.concatenate((self.line_ex_to_subid, trafo_ex_to_subid))

        # Get thermal limits of lines and trafos
        self.thermal_limit_a = np.concatenate(
            (self._lines_df.normamps, self._trafos_df.normamps)
        ).astype(dt_float)

        # TODO Map storages to substation ID
        self.set_no_storage()

        # Finish the initialization
        self._compute_pos_big_topo()

    def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:

        # Return if action is none
        if backendAction is None:
            return

        (active_sub, (prod_p, prod_v, load_p, load_q, storage), topo__, shunts__,) = backendAction()

        # Change the active values of the loads
        load_id2name = dict(zip(range(self.n_load), self.name_load))
        for load_id, new_p in load_p:
            self._grid.text(f"Edit Load.{load_id2name[load_id]} kw={new_p}")
        # Change the reactive values of the loads
        for load_id, new_q in load_q:
            self._grid.text(f"Edit Load.{load_id2name[load_id]} kvar={new_q}")

        # Change the active value of generators
        gen_id2name = dict(zip(range(self.n_gen), self.name_gen))
        for gen_id, new_p in prod_p:
            self._grid.text(f"Edit Generator.{gen_id2name[gen_id]} kw={new_p}")
        # Change the voltage value of generators (TODO discuss with Paulo)
        for gen_id, new_v in prod_v:
            self._grid.text(f"Edit Generator.{gen_id2name[gen_id]} kv={new_v}")

        # Handle the connection/disconnection of lines/trafos
        n_line_dss = self._lines_df.shape[0]
        line_id2name = dict(zip(list(range(self.n_line), map(str.lower, self.name_line))))
        # - on "or" side
        lines_or_bus = backendAction.get_lines_or_bus()
        for line_id, new_bus in lines_or_bus:
            el_type = "Line" if line_id < n_line_dss else "Transformer"
            enabled = "False" if new_bus == -1 else True
            self._grid.text(f"Edit {el_type}.{line_id2name[line_id]} enabled={enabled}")
        # - on "ex" side
        lines_ex_bus = backendAction.get_lines_ex_bus()
        for line_id, new_bus in lines_ex_bus:
            el_type = "Line" if line_id < n_line_dss else "Transformer"
            enabled = "False" if new_bus == -1 else True
            self._grid.text(f"Edit {el_type}.{line_id2name[line_id]} enabled={enabled}")

    def runpf(self, is_dc: dt_bool = False) -> Tuple[dt_bool, Union[Exception, None]]:

        try:
            if is_dc:   # TODO discuss with Paulo (later)
                # OpenDSS does not support DC power flow natively
                raise NotImplementedError("DC power flow is not supported in OpenDSS.")
            else:
                self._grid.text("Solve")
            if not self._grid.solution.converged:   # TODO works only for snapshot
                # Custom exception or just return False and a message/exception
                return False, RuntimeError("OpenDSS power flow did not converge.")
            return True, None
        except Exception as exc_:
            return False, exc_

    def _aux_get_topo_vect(self, res, dt, key, pos_topo_vect, add_id=0):
        # Loop through each element of the table
        # (each table representing either the loads, or the generators or
        # the powerlines or the trafos). Then we assign the right bus
        # (local - eg 1 or 2) to the right component of the vector "res"
        # (the component is given by the "pos_topo_vect" -
        # eg self.load_pos_topo_vect when we look at the loads)
        el_id = 0
        for (status, bus_id) in dt[["enabled", key]].values:
            my_pos_topo_vect = pos_topo_vect[el_id + add_id]
            bus_id = dt_int(bus_id.split("_")[1])
            if status == "true":
                local_bus = self.global_bus_to_local_int(bus_id, my_pos_topo_vect)
            else:
                local_bus = -1
            res[my_pos_topo_vect] = local_bus
            el_id += 1

    def get_topo_vect(self) -> np.ndarray:

        el_dfs = Elements(self._grid)
        res = np.full(self.dim_topo, fill_value=-2, dtype=int)

        # Topo of loads
        self._aux_get_topo_vect(res, el_dfs.loads_df, "bus1", self.load_pos_topo_vect)

        # Topo of generators
        self._aux_get_topo_vect(res, el_dfs.generators_df, "bus1", self.gen_pos_topo_vect)

        # Topo of each side of powerlines
        self._aux_get_topo_vect(res, el_dfs.lines_df, "bus1", self.line_or_pos_topo_vect)
        self._aux_get_topo_vect(res, el_dfs.lines_df, "bus2", self.line_ex_pos_topo_vect)

        # Topo of each side of transformers (appending to powerlines)
        n_line_dss = el_dfs.lines_df.shape[0]
        self._aux_get_topo_vect(
            res, el_dfs.transformers_df, "bus1", self.line_or_pos_topo_vect, add_id=n_line_dss
        )
        self._aux_get_topo_vect(
            res, el_dfs.transformers_df, "bus2", self.line_ex_pos_topo_vect, add_id=n_line_dss
        )

        return res

    def generators_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get active setpoint, reactive absorption / production and
        # voltage magnitude at the bus to which the gens are connected
        p_df, q_df = Powers(self._grid).powers_elements
        vmag_df, _ = Voltages(self._grid).voltages_elements
        gen_mask = p_df.index.map(lambda x: x.lower().split(".")[0]) == "generator"

        terminals = ["Terminal1.1", "Terminal1.2", "Terminal1.3"]

        prod_p = p_df[gen_mask][terminals].sum(axis=1).values.astype(dt_float)
        prod_q = q_df[gen_mask][terminals].sum(axis=1).values.astype(dt_float)
        prod_v = vmag_df[gen_mask][terminals].mean(axis=1).values.astype(dt_float)

        return prod_p, prod_q, prod_v

    def loads_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get active consumption, reactive consumption and
        # voltage magnitude of the bus to which loads are connected
        p_df, q_df = Powers(self._grid).powers_elements
        vmag_df, _ = Voltages(self._grid).voltages_elements
        load_mask = p_df.index.map(lambda x: x.lower().split(".")[0]) == "load"

        terminals = ["Terminal1.1", "Terminal1.2", "Terminal1.3"]

        load_p = p_df[load_mask][terminals].sum(axis=1).values.astype(dt_float)
        load_q = q_df[load_mask][terminals].sum(axis=1).values.astype(dt_float)
        load_v = vmag_df[load_mask][terminals].mean(axis=1).values.astype(dt_float)

        return load_p, load_q, load_v

    def lines_or_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get active flow, reactive flow, voltage magnitude of the bus
        # to which the line origin is connected and current flow
        p_df, q_df = Powers(self._grid).powers_elements
        vmag_df, _ = Voltages(self._grid).voltages_elements
        imag_df, _ = Currents(self._grid).currents_elements

        line_mask = p_df.index.map(lambda x: x.lower().split(".")[0]) == "line"
        trafo_mask = p_df.index.map(lambda x: x.lower().split(".")[0]) == "transformer"

        terminals_or = ["Terminal1.1", "Terminal1.2", "Terminal1.3"]

        line_p_or = p_df[line_mask][terminals_or].sum(axis=1).values.astype(dt_float)
        line_q_or = q_df[line_mask][terminals_or].sum(axis=1).values.astype(dt_float)
        line_v_or = vmag_df[line_mask][terminals_or].mean(axis=1).values.astype(dt_float)
        line_i_or = imag_df[line_mask][terminals_or].mean(axis=1).values.astype(dt_float)

        trafo_p_or = p_df[trafo_mask][terminals_or].sum(axis=1).values.astype(dt_float)
        trafo_q_or = q_df[trafo_mask][terminals_or].sum(axis=1).values.astype(dt_float)
        trafo_v_or = vmag_df[trafo_mask][terminals_or].mean(axis=1).values.astype(dt_float)
        trafo_i_or = imag_df[trafo_mask][terminals_or].mean(axis=1).values.astype(dt_float)

        line_p_or = np.concatenate((line_p_or, trafo_p_or))
        line_q_or = np.concatenate((line_q_or, trafo_q_or))
        line_v_or = np.concatenate((line_v_or, trafo_v_or))
        line_i_or = np.concatenate((line_i_or, trafo_i_or))

        return line_p_or, line_q_or, line_v_or, line_i_or

    def lines_ex_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get active flow, reactive flow, voltage magnitude of the bus
        # to which the line extreme is connected and current flow
        p_df, q_df = Powers(self._grid).powers_elements
        vmag_df, _ = Voltages(self._grid).voltages_elements
        imag_df, _ = Currents(self._grid).currents_elements

        line_mask = p_df.index.map(lambda x: x.lower().split(".")[0]) == "line"
        trafo_mask = p_df.index.map(lambda x: x.lower().split(".")[0]) == "transformer"

        terminals_ex = ["Terminal2.1", "Terminal2.2", "Terminal2.3"]

        # TODO confirm if the p_ex, q_ex == -p, -q (at terminal 2) in Pandapower
        line_p_ex = - p_df[line_mask][terminals_ex].sum(axis=1).values.astype(dt_float)
        line_q_ex = - q_df[line_mask][terminals_ex].sum(axis=1).values.astype(dt_float)
        line_v_ex = vmag_df[line_mask][terminals_ex].mean(axis=1).values.astype(dt_float)
        line_i_ex = imag_df[line_mask][terminals_ex].mean(axis=1).values.astype(dt_float)

        # TODO confirm if the p_ex, q_ex == -p, -q (at terminal 2) in Pandapower
        trafo_p_ex = - p_df[trafo_mask][terminals_ex].sum(axis=1).values.astype(dt_float)
        trafo_q_ex = - q_df[trafo_mask][terminals_ex].sum(axis=1).values.astype(dt_float)
        trafo_v_ex = vmag_df[trafo_mask][terminals_ex].mean(axis=1).values.astype(dt_float)
        trafo_i_ex = imag_df[trafo_mask][terminals_ex].mean(axis=1).values.astype(dt_float)

        line_p_ex = np.concatenate((line_p_ex, trafo_p_ex))
        line_q_ex = np.concatenate((line_q_ex, trafo_q_ex))
        line_v_ex = np.concatenate((line_v_ex, trafo_v_ex))
        line_i_ex = np.concatenate((line_i_ex, trafo_i_ex))

        return line_p_ex, line_q_ex, line_v_ex, line_i_ex
