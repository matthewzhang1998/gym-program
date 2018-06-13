import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='NumSwap-v0',
    entry_point='gym_program.envs:SwapEnv',
)

register(
    id='NumSort-v0',
    entry_point='gym_program.envs:SortEnv',
)

register(
    id='NumInsert-v0',
    entry_point='gym_program.envs:InsertEnv',
)

register(
    id='NumCopy-v0',
    entry_point='gym_program.envs:CopyEnv',
)

register(
    id='NumMax-v0',
    entry_point='gym_program.envs:MaxEnv',
)