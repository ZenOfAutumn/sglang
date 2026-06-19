import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.platforms import current_platform


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.

    中译：空闲睡眠器。调度器主循环若一直忙轮询（busy loop），在长时间无请求时会把
          CPU 线程钉死在 100%，既浪费功耗又抢占 CPU 散热余量（多 GPU 时每张卡各占一个
          线程尤为明显）。本类用 zmq.Poller 在「可能收到需立即处理数据」的套接字上阻塞
          等待，从而在空闲期让出 CPU；当有数据到达时立刻被唤醒。
    """

    def __init__(self, sockets):
        # 中译：创建 ZMQ 轮询器，并记录上次清空缓存的时间点。
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        # 中译：把所有传入的套接字都注册为「监听可读事件（POLLIN）」。
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        # 中译：周期性清空显存缓存的时间间隔（秒），<=0 表示不清空。
        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        # 中译：在已注册套接字上最多阻塞 1000ms 等待可读事件；
        #       期间若有数据到达会提前返回，否则超时返回——以此实现空闲让出 CPU。
        self.poller.poll(1000)
        # 中译：若开启了周期清缓存且距上次清空已超过间隔，则清空一次显存缓存并刷新计时。
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            current_platform.empty_cache()
