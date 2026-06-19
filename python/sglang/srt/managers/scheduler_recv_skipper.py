from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs


class SchedulerRecvSkipper:
    """按 forward mode 加权计数，控制调度器「收请求」的频率。

    中译：每轮 forward 不必都去 ZMQ 收一次新请求（收取本身有开销）。本类用一个累加计数器：
          按上一轮的 forward mode 取不同权重累加，达到阈值才放行一次「收请求」并清零。
          不同 mode 给不同权重，意味着某些轮（如 decode）推进得快/慢时可调节收取节奏。
    """

    @staticmethod
    def maybe_create(server_args: ServerArgs):
        # 中译：工厂方法——仅当 scheduler_recv_interval > 1（确实要跳过部分轮次）时才创建实例，否则返回 None。
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    def __init__(self, server_args: ServerArgs):
        # Can be supported if needed, but may need e.g. `global_forward_mode`
        # 中译：暂不支持 DP attention（如需支持，可能要引入 `global_forward_mode` 等全局态）。
        assert not server_args.enable_dp_attention
        self._counter = 0
        # 中译：阈值——计数累加到 >= 它时放行一次收取。
        self._threshold = server_args.scheduler_recv_interval
        # All can be tuned if needed
        # 中译：以下权重均可按需通过环境变量调优。_default_weight 用于未在表中列出的 forward mode。
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        # 中译：各 forward mode 对应的累加权重（DECODE / 投机的 TARGET_VERIFY / None 即非 forward）。
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def handle(self, last_forward_mode: ForwardMode):
        # 中译：根据上一轮的 forward mode 取权重累加到计数器；达到阈值则放行收取并清零。返回本轮是否应收。
        should_recv = False

        last_weight = self._weight_of_forward_mode.get(
            last_forward_mode, self._default_weight
        )
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv
