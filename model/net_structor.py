from collections import OrderedDict
import sys
sys.path.append('..')
from model.convlstmcell import ConvLSTM_cell

# encoder : convleaky in -> out -> clstm in  -> ...
# decoder : clstm_cell in -> out -> conv_leaky ...

convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [   
        ConvLSTM_cell(shape=(48,48), channels=16, kernel_size=5, features_num=64),
        ConvLSTM_cell(shape=(24,24), channels=64, kernel_size=5, features_num=96),
        ConvLSTM_cell(shape=(12,12), channels=96, kernel_size=5, features_num=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({                                       
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM_cell(shape=(12,12), channels=96, kernel_size=5, features_num=96),
        ConvLSTM_cell(shape=(24,24), channels=96, kernel_size=5, features_num=96),
        ConvLSTM_cell(shape=(48,48), channels=96, kernel_size=5, features_num=64),
    ]
]

convlstm_scs_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [   
        ConvLSTM_cell(shape=(668,668), channels=16, kernel_size=5, features_num=64),
        ConvLSTM_cell(shape=(334,334), channels=64, kernel_size=5, features_num=96),
        ConvLSTM_cell(shape=(167,167), channels=96, kernel_size=5, features_num=96)
    ]
]

convlstm_scs_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({                                       
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM_cell(shape=(167,167), channels=96, kernel_size=5, features_num=96),
        ConvLSTM_cell(shape=(334,334), channels=96, kernel_size=5, features_num=96),
        ConvLSTM_cell(shape=(668,668), channels=96, kernel_size=5, features_num=64),
    ]
]


fconvlstm_scs_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [   
        ConvLSTM_cell(shape=(668,668), channels=16, kernel_size=5, features_num=64,fconv=True),
        ConvLSTM_cell(shape=(334,334), channels=64, kernel_size=5, features_num=96,fconv=True),
        ConvLSTM_cell(shape=(167,167), channels=96, kernel_size=5, features_num=96,fconv=True)
    ]
]

fconvlstm_scs_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({                                       
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM_cell(shape=(167,167), channels=96, kernel_size=5, features_num=96,fconv=True),
        ConvLSTM_cell(shape=(334,334), channels=96, kernel_size=5, features_num=96,fconv=True),
        ConvLSTM_cell(shape=(668,668), channels=96, kernel_size=5, features_num=64,fconv=True),
    ]
]

fconvlstm_scs_min_cache_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 4, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [16, 16, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [24, 24, 3, 2, 1]}),
    ],

    [   
        ConvLSTM_cell(shape=(668,668), channels=4, kernel_size=5, features_num=16,fconv=True,frames_len=4,is_cuda=True),
        ConvLSTM_cell(shape=(334,334), channels=16, kernel_size=5, features_num=24,fconv=True,frames_len=4,is_cuda=True),
        ConvLSTM_cell(shape=(167,167), channels=24, kernel_size=5, features_num=24,fconv=True,frames_len=4,is_cuda=True)
    ]
]

fconvlstm_scs_min_cache_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [24, 24, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [24, 24, 4, 2, 1]}),
        OrderedDict({                                       
            'conv3_leaky_1': [16, 4, 3, 1, 1],
            'conv4_leaky_1': [4, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM_cell(shape=(167,167), channels=24, kernel_size=5, features_num=24,fconv=True,frames_len=4,is_cuda=True),
        ConvLSTM_cell(shape=(334,334), channels=24, kernel_size=5, features_num=24,fconv=True,frames_len=4,is_cuda=True),
        ConvLSTM_cell(shape=(668,668), channels=24, kernel_size=5, features_num=16,fconv=True,frames_len=4,is_cuda=True),
    ]
]


pconvlstm_scs_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [   
        ConvLSTM_cell(shape=(668,668), channels=16, kernel_size=5, features_num=64, fconv=False),
        ConvLSTM_cell(shape=(334,334), channels=64, kernel_size=5, features_num=96, fconv=False),
        ConvLSTM_cell(shape=(167,167), channels=96, kernel_size=5, features_num=96, fconv=False)
    ]
]

pconvlstm_scs_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({                                       
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM_cell(shape=(167,167), channels=96, kernel_size=5, features_num=96, fconv=False),
        ConvLSTM_cell(shape=(334,334), channels=96, kernel_size=5, features_num=96, fconv=False),
        ConvLSTM_cell(shape=(668,668), channels=96, kernel_size=5, features_num=64, fconv=False),
    ]
]

def generate_conv_lstm_encoder_decoder_params(frames_len=4, is_cuda=True):
    
    fconvlstm_scs_min_cache_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 4, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [16, 16, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [24, 24, 3, 2, 1]}),
        ],

        [   
            ConvLSTM_cell(shape=(668,668), channels=4, kernel_size=5, features_num=16,fconv=True,frames_len=frames_len,is_cuda=is_cuda),
            ConvLSTM_cell(shape=(334,334), channels=16, kernel_size=5, features_num=24,fconv=True,frames_len=frames_len,is_cuda=is_cuda),
            ConvLSTM_cell(shape=(167,167), channels=24, kernel_size=5, features_num=24,fconv=True,frames_len=frames_len,is_cuda=is_cuda)
        ]
    ]

    fconvlstm_scs_min_cache_decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [24, 24, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [24, 24, 4, 2, 1]}),
            OrderedDict({                                       
                'conv3_leaky_1': [16, 4, 3, 1, 1],
                'conv4_leaky_1': [4, 1, 1, 1, 0]
            }),
        ],

        [
            ConvLSTM_cell(shape=(167,167), channels=24, kernel_size=5, features_num=24,fconv=True,frames_len=frames_len,is_cuda=is_cuda),
            ConvLSTM_cell(shape=(334,334), channels=24, kernel_size=5, features_num=24,fconv=True,frames_len=frames_len,is_cuda=is_cuda),
            ConvLSTM_cell(shape=(668,668), channels=24, kernel_size=5, features_num=16,fconv=True,frames_len=frames_len,is_cuda=is_cuda),
            ]
        ]

    return fconvlstm_scs_min_cache_encoder_params, fconvlstm_scs_min_cache_decoder_params
