# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers.utils import logging

from ..configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class SupervisedModelConfig(PretrainedConfig):

    model_type = "supervisedmodel"

    def __init__(
        self,
        supervised_model_type=None,
        vocab_size=9,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.supervised_model_type = supervised_model_type
        self.vocab_size = vocab_size
