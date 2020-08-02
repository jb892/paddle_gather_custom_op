#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import paddle.fluid as fluid

file_dir = os.path.dirname(os.path.abspath(__file__))
fluid.load_op_library(os.path.join(file_dir, 'src/custom_lib.so'))

from paddle.fluid.layer_helper import LayerHelper

__all__ = ['gather_dim']

def gather_dim(input, index):
    """
    **Gather Dim Layer**
    Output is obtained by gathering entries of X indexed by `index`
    and concatenate them together.
    .. math::
        Out = X[Index]
    .. code-block:: text
        Given:
        X = [[1, 2, 3],
             [3, 4, 5],
             [5, 6, 7]]
        Index = [[1, 2]
        Then:
        Out = [[3, 4, 5],
               [5, 6, 7]]
    Args:
        input (Variable): The source input with rank>=1, This
                          is a 3-D tensor with shape of [B, N, 3].
        index (Variable): The index input with shape of [B, M, 3].

    Returns:
        output (Variable): The output is a tensor with shape of [B,M,3].
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 5, 3], dtype='float32')
            index = fluid.data(name='index', shape=[None, 1, 1], dtype='int32')
            output = fluid.layers.gather_point(x, index)
    """
    helper = LayerHelper('gather_dim', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather_dim",
        inputs={"X": input,
                "Index": index},
        outputs={"Output": out})
    return out
