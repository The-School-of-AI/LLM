"""Metrics server for P12 POC"""
import time
import yaml
import psutil
import threading
from prometheus_client import start_http_server, Gauge, Counter, Info

class MetricsServer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loss = Gauge('training_loss', 'Training loss')
        self.learning_rate = Gauge('learning_rate', 'Learning rate')
        self.throughput = Gauge('tokens_per_second', 'Training throughput')
        self.global_step = Gauge('global_step', 'Training step')
        self.gradient_norm = Gauge('gradient_norm', 'Gradient norm')
        self.checkpoint_saves = Counter('checkpoint_saves_total', 'Checkpoints saved')
        self.checkpoint_failures = Counter('checkpoint_failures_total', 'Checkpoint failures')
        self.last_checkpoint_time = Gauge('last_checkpoint_timestamp', 'Last checkpoint time')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage')
        self.training_status = Info('training_status', 'Training status')
        
        self.running = False
        self.collection_thread = None
        print("✓ MetricsServer initialized")
    
    def start(self):
        port = self.config['training']['metrics_port']
        start_http_server(port)
        print(f"✓ Metrics server started on port {port}")
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self.collection_thread.start()
    
    def _collect_system_metrics(self):
        while self.running:
            try:
                self.cpu_usage.set(psutil.cpu_percent(interval=1))
                self.memory_usage.set(psutil.virtual_memory().percent)
            except:
                pass
            time.sleep(5)
    
    def stop(self):
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        print("✓ Metrics server stopped")
    
    def update_training_metrics(self, loss, lr, step, tokens=None, grad_norm=None):
        self.loss.set(loss)
        self.learning_rate.set(lr)
        self.global_step.set(step)
        if grad_norm:
            self.gradient_norm.set(grad_norm)
    
    def update_throughput(self, tps):
        self.throughput.set(tps)
    
    def record_checkpoint(self, duration, success=True):
        if success:
            self.checkpoint_saves.inc()
            self.last_checkpoint_time.set(time.time())
        else:
            self.checkpoint_failures.inc()
    
    def update_training_status(self, status, message=""):
        self.training_status.info({'status': status, 'message': message})

_metrics_server = None

def get_metrics_server(config_path="config.yaml"):
    global _metrics_server
    if _metrics_server is None:
        _metrics_server = MetricsServer(config_path)
    return _metrics_server
