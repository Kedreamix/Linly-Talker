import time

# 定义装饰器
def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 运行时间： {end_time - start_time} 秒")
        return result
    return wrapper