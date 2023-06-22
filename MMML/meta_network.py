from __future__ import annotations  # allow to set return type to Self
from torch import nn
from copy import deepcopy
from inspect import signature

def test_compatibility(sig, inputs_names):
    remaining_input_names = [name for name in inputs_names]
    try:    
        for name, param in sig.parameters.items():
            if (param.kind == param.POSITIONAL_ONLY) or (param.kind == param.POSITIONAL_OR_KEYWORD):
                remaining_input_names.pop(0)
                
            elif (param.kind == param.VAR_POSITIONAL):
                #in this case, all the remaining values will be passed to args and kwargs will never be reached
                return 0
            elif param.kind == param.KEYWORD_ONLY:
                current_name = remaining_input_names.pop(0)
                assert name == current_name
            elif param.kind == param.VAR_KEYWORD:
                remaining_input_names.pop(0)

    except IndexError:
        raise Exception(f"invalid config, got : signature  {sig} and input name {inputs_names}")


class ForwardModule(nn.Module):
    def __init__(
        self,
        split_config: SplitConfigurationBuilder,
    ):
        super().__init__()

        # input handler is just a callable to unpack the incomming data
        self.ordered_nodes = split_config.get_config()
        
        # check that the configuration correspond to the given function
        for node in self.ordered_nodes:
            node_fn= node["node"]
            node_inputs = node["input_names"]
            assert callable(node_fn)
            try:
                sig = signature(node_fn)
                test_compatibility(sig, node_inputs)
            except ValueError:
                pass

            
            
        # register the modules for pytorch
        self.module = nn.ModuleList(
            node["node"]
            for node in self.ordered_nodes
            if issubclass(type(node["node"]), nn.Module)
        )

        # user can get losses at the end of the epoch
        self.list_metrics = [node for node in self.ordered_nodes if node["metric_name"]]

        number_nodes = len(self.ordered_nodes)
        list_input_names = [node["input_names"] for node in self.ordered_nodes]
        # define the tensor that can be freed while calling forward
        pop_keys = []
        for i, node in enumerate(self.ordered_nodes):
            current_input_to_pop = []
            
            # avoid error for list_input_names[(i+1):]
            if i == number_nodes - 1:
                pop_keys.append([])
                continue

            list_remaining_inputs = list_input_names[(i + 1) :]

            remaining_inputs = []
            for next_inputs in list_remaining_inputs:
                remaining_inputs += next_inputs

            for input in node["input_names"]:
                if input not in remaining_inputs:
                    current_input_to_pop.append(input)

            pop_keys.append(current_input_to_pop)
        self.pop_keys = pop_keys

    def forward(self, input):
        # heap : if a object isn't is the stack, it is not pointed by anything and is free to be collected
        # by garbage collection
        heap = {"input": input}
        loss = 0

        for instanciated_node_cfg, free_variables in zip(
            self.ordered_nodes, self.pop_keys
        ):
            input_names = instanciated_node_cfg["input_names"]
            output_names = instanciated_node_cfg["output_names"]
            node = instanciated_node_cfg["node"]

            current_inputs = [heap[input_name] for input_name in input_names]

            outputs = node(*current_inputs)

            if len(output_names) <= 1 and instanciated_node_cfg["is_loss"]:
                loss += outputs

            if len(output_names) == 1:
                output_name = output_names[0]
                heap[output_name] = outputs

            elif len(output_names) > 1:
                assert len(output_names) == len(
                    outputs
                ), "missmatch between output names : {output_names} and runtime outputs : {ouputs}" 
                assert not instanciated_node_cfg[
                    "is_loss"
                ], "only one output expected for loss"

                for output_name, output in zip(output_names, outputs):
                    heap[output_name] = output

            for name_to_free in free_variables:
                heap.pop(name_to_free)  # if keyerror : the pop_keys is not correct

        # here, if you are not using torch metric, put everything to a logger.
        # accumulate all losses/ metrics if necessary.
        return loss

    def accumulate_and_get_logs(self):
        """
        aggregate all metrics and return them, using torchmetrics
        """
        all_metrics = {}
        for node_config in self.list_metrics:
            node = node_config["node"]
            all_metrics[node_config["metric_name"]] = node.compute()
            node.reset()
        return all_metrics


class MultiDatasetBackbone(nn.Module):
    def __init__(self, dataset_input_to_backbone):
        self.dataset_input_to_backbone = nn.ModuleDict(dataset_input_to_backbone)

    def forward(self, input):
        input_name, input = input
        input_name = input_name.item()
        return self.dataset_input_to_backbone[input_name](input)


class SplitConfigurationBuilder:
    def __init__(self):
        self.config = []

    @classmethod
    def copy_config(cls, split_config: SplitConfigurationBuilder):
        new_instance = cls()
        new_instance.config = split_config.config  # do not copy the data
        return new_instance

    def _connect(
        self,
        node,
        input_names: list[str],
        output_names: list[str],
        is_loss,
        metric_name=None,
    ) -> SplitConfigurationBuilder:
        """
        metric name if it is a metric (if it iis not a metric, do not provide). Defaults to None.

        """
        self.config.append(
            dict(
                node=node,
                input_names=input_names,
                output_names=output_names,
                is_loss=is_loss,
                metric_name=metric_name,
            )
        )
        return self

    def connect_node(self, node, input_nams, output_names) -> SplitConfigurationBuilder:
        return self._connect(node, input_nams, output_names, False, None)

    def connect_loss(
        self, loss, input_names, metric_name=None
    ) -> SplitConfigurationBuilder:
        return self._connect(loss, input_names, [], True, metric_name)

    def connect_metric(
        self, metric, input_names, metric_name="_default_name_"
    ) -> SplitConfigurationBuilder:
        """
        Connect a metric
        If the name is not specified, default to the class name of the metric
        """
        if metric_name == "_default_name_":
            metric_name = str(metric)[:-2]
        return self._connect(metric, input_names, [], False, metric_name)

    def get_config(self):
        return self.config


def handle_classification_input(input):
    # unpack the input
    images, targets = input
    return images, targets
