# -*- coding: utf-8 -*-

"""Graph functions."""

from typing import Any, Dict, List, Union
import numpy as np
import onnx
from onnx import helper, numpy_helper, checker


class Graph:
    """ONNX graph manipulation class."""

    def __init__(self, model: onnx.ModelProto) -> None:
        """Create Graph."""
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(model)
        model = model_copy
        graph = model.graph
        self._init_dict = {
            x.name: x
            for x in graph.initializer
        }
        self._node_dict = {
            y: x
            for x in graph.node
            for y in x.output
        }
        self._name = graph.name
        self._nodes = list(graph.node)
        self._inputs = list(graph.input)
        self._outputs = list(graph.output)
        self._opset_imports = list(model.opset_import)
        self._model_info = {
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
        }

    @property
    def inputs(self) -> List[onnx.ValueInfoProto]:
        """Get list of inputs."""
        return self._inputs

    @property
    def outputs(self) -> List[onnx.ValueInfoProto]:
        """Get list of outputs."""
        return self._outputs

    @property
    def init_dict(self) -> Dict[str, onnx.ValueInfoProto]:
        """Get dictionary of initializers."""
        return self._init_dict

    @property
    def nodes(self) -> List[onnx.NodeProto]:
        """Get list of nodes."""
        return self._nodes

    def find_node_by_name(self, name: str) -> onnx.NodeProto:
        """Find node by name."""
        for node in self._nodes:
            if node.name == name:
                return node
        return None

    def find_node_by_output(self, name: str) -> onnx.NodeProto:
        """Find node by output."""
        return self._node_dict.get(name)

    def insert_node(self, node: onnx.NodeProto) -> None:
        """Insert new node into graph."""
        if self.find_node_by_name(node.name) is not None:
            raise ValueError(f"Node {node.name} already exists")
        self._nodes.append(node)
        for output in node.output:
            self._node_dict[output] = node

    def remove_node(self, node: onnx.NodeProto) -> None:
        """Remove node from graph."""
        self._nodes.remove(node)
        for output in node.output:
            if self._node_dict[output] is node:
                del self._node_dict[output]

    def create_node(self, name: str, op_type: str, inputs: List[str] = None,
                    outputs: List[str] = None, **kwargs) -> onnx.NodeProto:
        """Create new node."""
        if "domain" in kwargs:
            domain = kwargs["domain"]
            for opset in self._opset_imports:
                if opset.domain == domain:
                    break
            else:
                self._opset_imports.append(
                    helper.make_operatorsetid(domain, 1)
                )
        node = helper.make_node(
            op_type,
            inputs=inputs or [],
            outputs=outputs or [name],
            name=name,
            **kwargs
        )
        self.insert_node(node)
        return node

    def create_or_replace_node(self, name: str, *args,
                               **kwargs) -> onnx.NodeProto:
        """Create or replace node."""
        old_node = self.find_node_by_name(name)
        if old_node is not None:
            self.remove_node(old_node)
        return self.create_node(name, *args, **kwargs)

    def create_constant(self, name: str, value: Any) -> onnx.NodeProto:
        """Create constant node."""
        return self.create_node(
            name=name,
            op_type="Constant",
            value=numpy_helper.from_array(
                arr=np.array(value),
                name=name
            )
        )

    def create_value(self, name: str, value: Any) -> None:
        """Create value."""
        self._init_dict[name] = numpy_helper.from_array(
            arr=np.array(value),
            name=name
        )

    def serialize(
        self,
        inputs: Union[None, List[onnx.ValueInfoProto]] = None,
        outputs: Union[None, List[onnx.ValueInfoProto]] = None,
    ) -> onnx.ModelProto:
        """Serialize graph to proto."""
        if inputs is None:
            inputs = self._inputs
        if outputs is None:
            outputs = self._outputs
        nodes = []
        inits = []
        visited_nodes = set()
        visited_inits = set()

        def traverse(output):
            if output in self._init_dict:
                if output in visited_inits:
                    return
                visited_inits.add(output)
                inits.append(self._init_dict[output])
                return
            if output not in self._node_dict:
                return
            node = self._node_dict[output]
            if node.name in visited_nodes:
                return
            visited_nodes.add(node.name)
            for inp in node.input:
                traverse(inp)
            nodes.append(node)

        for out in outputs:
            traverse(out.name)

        graph = helper.make_graph(
            nodes=nodes,
            name=self._name,
            inputs=inputs,
            outputs=outputs,
            initializer=inits,
        )
        model = helper.make_model(
            graph,
            opset_imports=self._opset_imports,
            **self._model_info
        )
        checker.check_model(model, full_check=True)
        return model
