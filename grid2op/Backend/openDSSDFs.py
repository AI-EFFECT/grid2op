from typing import Tuple
import pandas as pd
import py_dss_interface as DSS


class Elements:
    def __init__(self, dss: DSS):
        self._dss = dss

    @property
    def lines_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.lines)

    @property
    def transformers_df(self) -> pd.DataFrame:
        trafos_df = self.__create_element_dataframe(self._dss.transformers)
        trafos_bus1, trafos_bus2 = [], []
        for trafo_buses in trafos_df.buses:
            trafos_bus1.append(trafo_buses.strip("[]").split(",")[0])
            trafos_bus2.append(trafo_buses.strip("[]").split(",")[1].strip(" "))
        trafos_df["bus1"], trafos_df["bus2"] = trafos_bus1, trafos_bus2
        return trafos_df

    @property
    def meters_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.meters)

    @property
    def monitors_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.monitors)

    @property
    def generators_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.generators)

    @property
    def vsources_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.vsources)

    @property
    def regcontrols_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.regcontrols)

    @property
    def loads_df(self) -> pd.DataFrame:
        return self.__create_element_dataframe(self._dss.loads)

    def __create_element_dataframe(self, element):
        if element.count == 0:
            return None

        element.first()
        element_properties = self._dss.cktelement.property_names

        dict_to_df = dict()

        name_list = list()

        for element_name in element.names:
            element.name = element_name
            if self._dss.cktelement.is_enabled:
                name_list.append(element.name.lower())
        dict_to_df["name"] = name_list

        for element_property in element_properties:
            property_list = list()

            for element_name in element.names:
                element.name = element_name
                if self._dss.cktelement.is_enabled:
                    property_list.append(
                        self._dss.dssproperties.value_read(
                            str(self._dss.cktelement.property_names.index(element_property) + 1)))

            dict_to_df[element_property.lower()] = property_list

        return pd.DataFrame().from_dict(dict_to_df)


class Powers:
    def __init__(self, dss: DSS):
        self._dss = dss

    @property
    def powers_elements(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.__create_dataframe()

    def __create_dataframe(self):
        node_order = [node.lower() for node in self._dss.circuit.y_node_order]
        element_nodes = dict()
        element_p = dict()
        element_q = dict()
        elements = list()

        is_there_pd = self._dss.circuit.pd_element_first()
        while is_there_pd:
            element = self._dss.cktelement.name.lower()
            num_phases = self._dss.cktelement.num_phases
            num_terminals = self._dss.cktelement.num_terminals
            num_conductors = self._dss.cktelement.num_conductors

            nodes = self.__create_terminal_list(self._dss.cktelement.node_order, num_terminals)
            p = self._dss.cktelement.powers[: 2 * num_terminals * num_conductors: 2]
            q = self._dss.cktelement.powers[1: 2 * num_terminals * num_conductors: 2]

            element_nodes[element] = nodes
            element_p[element] = p
            element_q[element] = q
            elements.append(element)

            if not self._dss.circuit.pd_element_next():
                is_there_pd = False

        is_there_pc = self._dss.circuit.pc_element_first()
        while is_there_pc:
            element = self._dss.cktelement.name.lower()
            num_phases = self._dss.cktelement.num_phases
            num_terminals = self._dss.cktelement.num_terminals
            num_conductors = self._dss.cktelement.num_conductors

            nodes = self.__create_terminal_list(self._dss.cktelement.node_order, num_terminals)
            p = self._dss.cktelement.powers[: 2 * num_terminals * num_conductors: 2]
            q = self._dss.cktelement.powers[1: 2 * num_terminals * num_conductors: 2]

            element_nodes[element] = nodes
            element_p[element] = p
            element_q[element] = q
            elements.append(element)

            if not self._dss.circuit.pc_element_next():
                is_there_pc = False

        p_df = pd.DataFrame(index=elements)

        for element, nodes in element_nodes.items():
            for order, node in enumerate(nodes):
                # column_name = f'node{node}'
                p_df.loc[element, node] = element_p[element][order]

        q_df = pd.DataFrame(index=elements)

        for element, nodes in element_nodes.items():
            for order, node in enumerate(nodes):
                # column_name = f'node{node}'
                q_df.loc[element, node] = element_q[element][order]

        return p_df, q_df

    # TODO move around
    def __create_terminal_list(self, nodes, num_terminals):
        terminal_list = []
        for i, node in enumerate(nodes):
            terminal_number = int((i // (len(nodes) / num_terminals))) + 1
            terminal_list.append(f'Terminal{terminal_number}.{node}')

        return terminal_list


class Voltages:
    def __init__(self, dss: DSS):
        self._dss = dss

    @property
    def voltages_elements(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.__create_dataframe()

    def __create_dataframe(self):
        node_order = [node.lower() for node in self._dss.circuit.y_node_order]
        element_nodes = dict()
        element_vmags = dict()
        element_vangs = dict()
        elements = list()

        is_there_pd = self._dss.circuit.pd_element_first()
        while is_there_pd:
            element = self._dss.cktelement.name.lower()
            num_phases = self._dss.cktelement.num_phases
            num_terminals = self._dss.cktelement.num_terminals
            num_conductors = self._dss.cktelement.num_conductors

            nodes = self.__create_terminal_list(self._dss.cktelement.node_order, num_terminals)
            vmags = self._dss.cktelement.voltages_mag_ang[: 2 * num_terminals * num_conductors: 2]
            vangs = self._dss.cktelement.voltages_mag_ang[1: 2 * num_terminals * num_conductors: 2]

            bus1, bus2 = self._dss.cktelement.bus_names[0].split(".")[0].lower(), \
                self._dss.cktelement.bus_names[1].split(".")[0].lower()

            self._dss.circuit.set_active_bus(bus1)
            kv_base1 = self._dss.bus.kv_base * 1000.0

            self._dss.circuit.set_active_bus(bus2)
            kv_base2 = self._dss.bus.kv_base * 1000.0

            for i in range(int(len(vmags) / 2)):
                vmags[i] = vmags[i] / kv_base1

            for i in range(int(len(vmags) / 2), len(vmags)):
                vmags[i] = vmags[i] / kv_base2

            element_nodes[element] = nodes
            element_vmags[element] = vmags
            element_vangs[element] = vangs
            elements.append(element)

            if not self._dss.circuit.pd_element_next():
                is_there_pd = False

        is_there_pc = self._dss.circuit.pc_element_first()
        while is_there_pc:
            element = self._dss.cktelement.name.lower()
            num_phases = self._dss.cktelement.num_phases
            num_terminals = self._dss.cktelement.num_terminals
            num_conductors = self._dss.cktelement.num_conductors

            nodes = self.__create_terminal_list(self._dss.cktelement.node_order, num_terminals)
            vmags = self._dss.cktelement.voltages_mag_ang[: 2 * num_terminals * num_conductors: 2]
            vangs = self._dss.cktelement.voltages_mag_ang[1: 2 * num_terminals * num_conductors: 2]

            bus1 = self._dss.cktelement.bus_names[0].split(".")[0].lower()

            self._dss.circuit.set_active_bus(bus1)
            kv_base1 = self._dss.bus.kv_base * 1000.0

            for i in range(len(vmags)):
                vmags[i] = vmags[i] / kv_base1

            element_nodes[element] = nodes
            element_vmags[element] = vmags
            element_vangs[element] = vangs
            elements.append(element)

            if not self._dss.circuit.pc_element_next():
                is_there_pc = False

        vmags_df = pd.DataFrame(index=elements)

        for element, nodes in element_nodes.items():
            for order, node in enumerate(nodes):
                # column_name = f'node{node}'
                vmags_df.loc[element, node] = element_vmags[element][order]

        vangs_df = pd.DataFrame(index=elements)

        for element, nodes in element_nodes.items():
            for order, node in enumerate(nodes):
                # column_name = f'node{node}'
                vangs_df.loc[element, node] = element_vangs[element][order]

        return vmags_df, vangs_df

    # TODO move around
    def __create_terminal_list(self, nodes, num_terminals):
        terminal_list = []
        for i, node in enumerate(nodes):
            terminal_number = int((i // (len(nodes) / num_terminals))) + 1
            terminal_list.append(f'Terminal{terminal_number}.{node}')

        return terminal_list


# -*- coding: utf-8 -*-
# @Author  : Paulo Radatz
# @Email   : paulo.radatz@gmail.com

from typing import Tuple

import pandas as pd
from py_dss_interface import DSS


class Currents:
    def __init__(self, dss: DSS):
        self._dss = dss

    @property
    def currents_elements(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.__create_dataframe()

    def __create_dataframe(self):
        node_order = [node.lower() for node in self._dss.circuit.y_node_order]
        element_nodes = dict()
        element_imags = dict()
        element_iangs = dict()
        elements = list()

        is_there_pd = self._dss.circuit.pd_element_first()
        while is_there_pd:
            element = self._dss.cktelement.name.lower()
            num_phases = self._dss.cktelement.num_phases
            num_terminals = self._dss.cktelement.num_terminals
            num_conductors = self._dss.cktelement.num_conductors

            nodes = self.__create_terminal_list(self._dss.cktelement.node_order, num_terminals)
            imags = self._dss.cktelement.currents_mag_ang[: 2 * num_terminals * num_conductors: 2]
            iangs = self._dss.cktelement.currents_mag_ang[1: 2 * num_terminals * num_conductors: 2]

            element_nodes[element] = nodes
            element_imags[element] = imags
            element_iangs[element] = iangs
            elements.append(element)

            if not self._dss.circuit.pd_element_next():
                is_there_pd = False

        is_there_pc = self._dss.circuit.pc_element_first()
        while is_there_pc:
            element = self._dss.cktelement.name.lower()
            num_phases = self._dss.cktelement.num_phases
            num_terminals = self._dss.cktelement.num_terminals
            num_conductors = self._dss.cktelement.num_conductors

            nodes = self.__create_terminal_list(self._dss.cktelement.node_order, num_terminals)
            imags = self._dss.cktelement.currents_mag_ang[: 2 * num_terminals * num_conductors: 2]
            iangs = self._dss.cktelement.currents_mag_ang[1: 2 * num_terminals * num_conductors: 2]

            element_nodes[element] = nodes
            element_imags[element] = imags
            element_iangs[element] = iangs
            elements.append(element)

            if not self._dss.circuit.pc_element_next():
                is_there_pc = False

        imags_df = pd.DataFrame(index=elements)

        for element, nodes in element_nodes.items():
            for order, node in enumerate(nodes):
                # column_name = f'node{node}'
                imags_df.loc[element, node] = element_imags[element][order]

        iangs_df = pd.DataFrame(index=elements)

        for element, nodes in element_nodes.items():
            for order, node in enumerate(nodes):
                # column_name = f'node{node}'
                iangs_df.loc[element, node] = element_iangs[element][order]

        return imags_df, iangs_df

    def __create_terminal_list(self, nodes, num_terminals):
        terminal_list = []
        for i, node in enumerate(nodes):
            terminal_number = int((i // (len(nodes) / num_terminals))) + 1
            terminal_list.append(f'Terminal{terminal_number}.{node}')

        return terminal_list
