# Copyright (c) Microsoft. All rights reserved.

"""This package contains a *hacky* integration of VERL with Agent Lightning."""

from agentlightning._transformers_compat import patch_transformers_vision2seq_alias

patch_transformers_vision2seq_alias()

from agentlightning.verl.compat import patch_flash_attn_padding_fallback

patch_flash_attn_padding_fallback()

from .daemon import *
from .dataset import *
from .entrypoint import *
from .trainer import *
