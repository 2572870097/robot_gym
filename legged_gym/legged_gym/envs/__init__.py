# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .go2.go2_constraint_robot import Go2ConstraintRobot
from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO
from legged_gym.envs.go2.go2_constraint_him import Go2ConstraintHimRoughCfg, Go2ConstraintHimRoughCfgPPO
from legged_gym.envs.go2.go2_backflip_config import Go2BackflipCfg, Go2BackflipCfgPPO
from legged_gym.envs.go2.go2_backflip_robot import Go2Backflip



from legged_gym.envs.yun1.yun1_config import Yun1RoughCfg, Yun1RoughCfgPPO

from legged_gym.envs.yun1w.yun1w_robot import Yun1w
from legged_gym.envs.yun1w.yun1w_config import Yun1WRoughCfg, Yun1WRoughCfgPPO
from legged_gym.envs.yun1w.yun1w_rough_config import Yun1WRoughCfg1, Yun1WRoughCfgPPO1
from legged_gym.envs.yun1w.yun1w_bipe_config import Yun1WBipeCfg, Yun1WBipeCfgPPO
from legged_gym.envs.yun1w.yun1w_trot_config import Yun1WTrotCfg, Yun1WTrotCfgPPO
from legged_gym.envs.yun1w.yun1w_jump_config import Yun1WJumpCfg, Yun1WJumpCfgPPO
import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, Go2RoughCfg(), Go2RoughCfgPPO() )
task_registry.register( "go2_np3o", Go2ConstraintRobot, Go2ConstraintHimRoughCfg(), Go2ConstraintHimRoughCfgPPO() )
task_registry.register( "go2_backflip", Go2Backflip, Go2BackflipCfg(), Go2BackflipCfgPPO() )
task_registry.register( "yun1", LeggedRobot, Yun1RoughCfg(), Yun1RoughCfgPPO() )
task_registry.register( "yun1w", Yun1w, Yun1WRoughCfg(), Yun1WRoughCfgPPO() )
task_registry.register( "yun1w_rough", Yun1w, Yun1WRoughCfg1(), Yun1WRoughCfgPPO1() )
task_registry.register( "yun1w_bipe", Yun1w, Yun1WBipeCfg(), Yun1WBipeCfgPPO() )
task_registry.register( "yun1w_trot", Yun1w, Yun1WTrotCfg(), Yun1WTrotCfgPPO() )
task_registry.register( "yun1w_jump", Yun1w, Yun1WJumpCfg(), Yun1WJumpCfgPPO() )