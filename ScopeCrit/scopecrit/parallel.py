"""有界样本线程池；单条样本内部继续由 Engine 按顺序执行。"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def bounded_results(function, items, workers, should_stop=lambda result: False):
    """至多 workers 个未完成任务；遇到停止条件后只收尾已启动任务。"""
    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scopecrit")
    pending = {}
    stopped = False

    def fill():
        while not stopped and len(pending) < workers:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(function, item)] = item

    try:
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            completed = []
            for future in done:
                item = pending.pop(future)
                result = future.result()
                stopped |= should_stop(result)
                completed.append((item, result))
            for item, result in completed:
                yield item, result
            fill()
    finally:
        # 先等待在途请求完成并保存，再让调用者关闭数据库。
        executor.shutdown(wait=True, cancel_futures=True)
