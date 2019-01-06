from keras.layers import Conv2D
from keras.layers import Activation, Input, Concatenate, Lambda
from keras.models import Model
from keras import backend as K

from .blocks import DecoderBlock
from ..utils import get_layer_number, to_tuple


def prep_classes(x, n_types=28, conv_size=7):
    x = K.repeat_elements(x, conv_size**2, 1)
    x = K.reshape(x, shape=(-1, n_types, conv_size, conv_size))
    x = K.permute_dimensions(x, [0,2,3,1])
    return x

def build_linknet(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    # here we merge backbone output and class input
    class_input = Input(shape=(28,), name='class_input')
    x_ = Lambda(prep_classes, name='class_lambda')(class_input)
    x = Concatenate(-1, name='class_concat')([x, x_])

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(inputs=[input, class_input], output=x)

    return model # could also return backbone for transfer learning
