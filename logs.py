import csv
import time
import psutil
import GPUtil

def log_performance(log_file='performance_log.txt', interval=1):
    with open(log_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'timestamp',
            'gpu_memory_utilization',
            'gpu_power_usage',
            'gpu_temp',
            'system_memory_utilization',
            'cpu_utilization',
            'disk_utilization',
            'system_memory_usage'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                # Get GPU details
                gpus = GPUtil.getGPUs()
                gpu = gpus[0] if gpus else None

                # Get system details
                cpu_utilization = psutil.cpu_percent(interval=None)
                system_memory = psutil.virtual_memory()
                disk_utilization = psutil.disk_usage('/').percent

                # Log details
                log_data = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'gpu_memory_utilization': getattr(gpu, 'memoryUtil', 'N/A'),
                    'gpu_power_usage': f"{getattr(gpu, 'powerDraw', 'N/A')} W" if gpu else 'N/A',
                    'gpu_temp': f"{getattr(gpu, 'temperature', 'N/A')} ℃" if gpu else 'N/A',
                    'system_memory_utilization': system_memory.percent,
                    'cpu_utilization': cpu_utilization,
                    'disk_utilization': disk_utilization,
                    'system_memory_usage': system_memory.used / system_memory.total
                }
                writer.writerow(log_data)
                print(log_data)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("Monitoring stopped.")



if __name__ == "__main__":
    log_performance()
