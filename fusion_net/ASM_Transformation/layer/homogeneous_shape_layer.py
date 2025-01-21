# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
from .shape_layer import ShapeLayer
from .homogeneous_transform_layer import HomogeneousTransformationLayer

class HomogeneousShapeLayer(torch.nn.Module):
    """
    Module to Perform a Shape Prediction
    (including a global homogeneous transformation)
    """
    def __init__(self, n_dims, n_evecs=0):
        """
        Parameters
        ----------
        shapes : np.ndarray
            shapes to construct a :class:`ShapeLayer`
        n_dims : int
            number of shape dimensions
        use_cpp : bool
            whether or not to use (experimental) C++ Implementation

        See Also
        --------
        :class:`ShapeLayer`
        :class:`HomogeneousTransformationLayer`
        """
        super().__init__()
        self._shape_layer = ShapeLayer(n_evecs)                         
        self._homogen_trafo = HomogeneousTransformationLayer(n_dims)   

        self.register_buffer("_indices_shape_params",
                             torch.arange(self._shape_layer.num_params))   # Example: arrange(5) = tensor([0, 1, 2, 3, 4])
        self.register_buffer("_indices_homogen_params",
                             torch.arange(self._shape_layer.num_params,    # Example: arrange(5, 9) = tensor([5, 6, 7, 8])
                                          self.num_params)) 

    def forward(self, params: torch.Tensor, mean, component):
        """
        Performs the actual prediction

        Parameters
        ----------
        params : :class:`torch.Tensor`
            input parameters

        Returns
        -------
        :class:`torch.Tensor`
            predicted shape
        """
        # print(params.shape)  # torch.Size([n_img, num_shape_params + num_Transformation_params, 1, 1])   
        shape_params = params.index_select(                         # torch.Size([n_img, num_shape_params, 1, 1])   
            dim=1, index=getattr(self, "_indices_shape_params")
        )           
        # print(shape_params.shape)
        transformation_params = params.index_select(                # torch.Size([n_img, num_Transformation_params, 1, 1])  
            dim=1, index=getattr(self, "_indices_homogen_params")
        )
        # print(transformation_params.shape)
        
        ############# First find the shape, then apply affine transformation (original code)
        shapes = self._shape_layer(shape_params, mean, component)   # Generate shape based on eigenvector P and (shape_params: b) + mean_shape(shape[0])
    
        transformed_shapes = self._homogen_trafo(shapes, transformation_params)  # Apply affine transformation to get the final shape

        return transformed_shapes

    @property
    def num_params(self):
        """
        Property to access this layer's number of parameters

        Returns
        -------
        int
            number of parameters
        """
        return self._shape_layer.num_params + self._homogen_trafo.num_params
