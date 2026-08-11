
def get_param_tree(tree, is_root=False):
    """
    Extract parameter tree from a trained tree structure for model serialization.
    
    This function recursively extracts essential parameters from a trained tree structure,
    creating a serializable representation that contains only the necessary information
    for model inference and state restoration. It handles both leaf nodes (containing
    trained models) and internal nodes (containing split information).
    
    Parameters
    ----------
    tree : dict
        The tree structure to extract parameters from. Expected to have:
        - 'type': Either 'leaf' or 'node' indicating the node type
        - For leaf nodes: 'model' containing the trained RFM model
        - For internal nodes: 'split_direction', 'split_point', 'left', 'right'
        - 'train_indices': Training sample indices that reached this node
        
    is_root : bool, default=False
        Whether this is the root node of the tree. Used for tree structure
        identification and potential special handling of root node parameters.
    
    Returns
    -------
    dict
        Parameter tree containing serialized model parameters:
        
        For leaf nodes:
        - 'type': 'leaf'
        - 'bandwidth': Kernel bandwidth parameter from the leaf model
        - 'weights': Model weights (alpha coefficients) from kernel regression
        - 'M': Mahalanobis matrix for feature space transformation
        - 'sqrtM': Square root of Mahalanobis matrix (if used by kernel)
        - 'train_indices': Training sample indices for this leaf
        - 'is_root': Boolean indicating if this is the root node
        
        For internal nodes:
        - 'type': 'node'
        - 'split_direction': Feature space direction vector used for splitting
        - 'split_point': Threshold value for the split decision
        - 'adaptive_temp_scaling': Inter-quartile range used to scale temperature at this node
        - 'left': Parameter tree for left child (recursively extracted)
        - 'right': Parameter tree for right child (recursively extracted)
        - 'is_root': Boolean indicating if this is the root node
    
    Examples
    --------
    >>> # Extract parameters from a trained tree for serialization
    >>> param_tree = get_param_tree(trained_tree, is_root=True)
    >>> 
    >>> # The resulting parameter tree can be used for model saving/loading
    >>> # or for analyzing the structure of the trained model
    >>> 
    >>> # For a leaf node, access model parameters:
    >>> if param_tree['type'] == 'leaf':
    ...     bandwidth = param_tree['bandwidth']
    ...     weights = param_tree['weights']
    ...     M_matrix = param_tree['M']
    >>> 
    >>> # For internal nodes, traverse the tree structure:
    >>> if param_tree['type'] == 'node':
    ...     left_subtree = param_tree['left']
    ...     right_subtree = param_tree['right']
    ...     split_direction = param_tree['split_direction']
    """
    if tree['type'] == 'leaf':
        leaf_model = tree['model']
        param_tree = {
            'type': 'leaf',
            'bandwidth': leaf_model.kernel_obj.bandwidth,
            'weights': leaf_model.weights,
            'M': leaf_model.M,
            'sqrtM': leaf_model.sqrtM,
            'train_indices': tree['train_indices'],
            'is_root': is_root
        }
        return param_tree
    else:
        return {
            'type': 'node',
            'split_direction': tree['split_direction'],
            'split_point': tree['split_point'],
            'adaptive_temp_scaling': tree.get('adaptive_temp_scaling', 1.0),
            'left': get_param_tree(tree['left'], is_root=False),
            'right': get_param_tree(tree['right'], is_root=False),
            'is_root': is_root
        }
