# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch


class ShapeLayer(torch.nn.Module):
    """
    Wrapper to compine Python and C++ Implementation under Single API

    """
    def __init__(self,n_evecs = 0):
        """

        Parameters
        ----------
        shapes : np.ndarray
            the actual shape components
        use_cpp : bool
            whether or not to use the (experimental) C++ Implementation
        """
        super().__init__()

        self._layer = _ShapeLayer(n_evecs)

    def forward(self, shape_params: torch.Tensor,mean,component):
        """
        Forwards parameters to Python or C++ Implementation

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            parameters for shape ensembling

        Returns
        -------
        :class:`torch.Tensor`
            Ensempled Shape

        """
        return self._layer(shape_params,mean,component)

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return self._layer.num_params


class _ShapeLayer(torch.nn.Module):
    """
    Python Implementation of Shape Layer

    """
    def __init__(self,n_evecs):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()

        self.n_evecs = n_evecs
    def forward(self, shape_params: torch.Tensor,mean,components):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            shape parameters

        Returns
        -------
        :class:`torch.Tensor`
            ensembled shape

        """
        components = components.expand(shape_params.size(0), #shape_params.size(0): num_img  *components.size()[1:]:num_shape_parames,num_point,2   1,5,9,2
                                       *components.size()[1:])
        
        weighted_components = components.mul(               #  shape_params:1,5,1,1 5个特征向量的参数(同b的t个参数)
            shape_params.expand_as(components))             # 等价于 P*b   P:5*(9*2)   b:1,5,1,1->1,5,9,2
        
        shapes = mean.add(weighted_components.sum(dim=1))  # shape = _shape_mean((1,9,2)) + P*b(1,9,2)
        return shapes

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        # return getattr(self, "_shape_components").size(1)
        return self.n_evecs-1

