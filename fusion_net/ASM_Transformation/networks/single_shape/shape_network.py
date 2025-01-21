# The code in the ASM_Transformation folder is adapted from: https://github.com/justusschock/shapenet

import torch
import logging
from ..feature_extractors import DCM
from ..abstract_network import AbstractShapeNetwork
logger = logging.getLogger(__file__)

class ShapeNetwork(AbstractShapeNetwork):
    """
    Network to Predict a single shape
    """

    def __init__(self, layer_cls,
                 n_dims = 2,
                 use_cpp = False,
                 in_channels=1,
                 norm_type='instance',
                 img_size=224,
                 feature_extractor=None,
                 n_evecs=0,
                 **kwargs,
                 ):
        """

        Parameters
        ----------
        layer_cls :
            Class to instantiate the last layer (usually a shape-constrained
            or transformation layer)
        layer_kwargs : dict
            keyword arguments to create an instance of `layer_cls`
        in_channels : int
            number of input channels
        norm_type : string or None
            Indicates the type of normalization used in this network;
            Must be one of [None, 'instance', 'batch', 'group']
        kwargs :
            additional keyword arguments

        """

        super().__init__(layer_cls=layer_cls,   #layer_cls = HomogeneousShapeLayer
                         n_dims = 2,
                         in_channels=in_channels,
                         norm_type=norm_type,
                         img_size=img_size,
                         feature_extractor=feature_extractor,
                         **kwargs)
        self._kwargs = kwargs
        self.n_channels = in_channels
        self._out_layer = layer_cls(n_dims = n_dims,n_evecs = n_evecs)
        self.num_out_params = n_evecs + 4     #self._out_layer.num_params   #  num_shape_params: 5  + transformation_params:4
        self.img_size = img_size
        self._model = DCM(num_out_params = self.num_out_params,input_channels=3)
        
        
    def forward(self, input_images,mean,components):
        """
        Forward input batch through network and shape layer

        Parameters
        ----------
        input_images : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
            predicted shapes

        """
        
        ##only lmks
        out_params,features_encoder,seg_neck = self._model(input_images)    
        
        lmks = self._out_layer(out_params.view(input_images.size(0),
                                             self.num_out_params, 1, 1),mean,components)
        
        return lmks,features_encoder,seg_neck
    


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        if isinstance(model, torch.nn.Module):
            self._model = model
        else:
            raise AttributeError("Invalid Model")

    @staticmethod
    def closure(model, data_dict: dict,
                optimizers: dict, criterions={}, metrics={},
                fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model : :class:`ShapeNetwork`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        criterions : dict
            dict holding the criterions to calculate errors
            (gradients from different criterions will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict criterions)
        list
            Arbitrary number of predictions as :class:`torch.Tensor`

        Raises
        ------
        AssertionError
            if optimizers or criterions are empty or the optimizers are not
            specified
        """
        
        if not criterions:
            criterions = kwargs.pop('losses', {})

        assert (optimizers and criterions) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict.pop("data")
            preds = model(inputs)

            if data_dict:

                for key, crit_fn in criterions.items():
                    _loss_val = crit_fn(preds["pred"], *data_dict.values())
                    loss_vals[key] = _loss_val.detach()
                    total_loss += _loss_val

                with torch.no_grad():
                    for key, metric_fn in metrics.items():
                        metric_vals[key] = metric_fn(
                            preds["pred"], *data_dict.values())

        if optimizers:
            optimizers['default'].zero_grad()
            total_loss.backward()
            optimizers['default'].step()

        else:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})
            
        for key, val in metric_vals.items():
            if isinstance(val, torch.Tensor):
                metric_vals[key] = val.detach().cpu().numpy()
                
        for key, val in loss_vals.items():
            if isinstance(val, torch.Tensor):
                loss_vals[key] = val.detach().cpu().numpy()

        return metric_vals, loss_vals, {k: v.detach() 
                                        for k, v in preds.items()}
