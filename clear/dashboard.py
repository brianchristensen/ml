# dashboard.py
from rich.console import Console
from rich.table import Table
from rich.text import Text
import time

class TrainingDashboard:
    def __init__(self):
        self.console = Console()
        self.global_history = []
        self.node_snapshots = []
        self.start_time = None

    def start(self, start_time):
        self.start_time = start_time

    def new_node(self, node_index, epoch, acc_at_start):
        self.node_snapshots.append({
            "Node": node_index,
            "Start Epoch": epoch,
            "Start Acc": acc_at_start,
            "Current Acc": acc_at_start,
            "Gate Entropy": None,
            "ProtoDiv": None,
            "Recon": None,
            "Usage": None,
            "Variance": None,
            "Temp": None
        })

    def update_node_metrics(self, node_idx, acc, gate_entropy, proto_div, recon, temp, node_div=None):
        node = self.node_snapshots[node_idx]
        node["Current Acc"] = acc
        node["ProtoDiv"] = proto_div
        node["NodeDiv"] = node_div
        node["Recon"] = recon
        node["Temp"] = temp
        node["Gate Entropy"] = gate_entropy
            
    def log_epoch(self, epoch, acc, loss):
        self.global_history.append({
            "Epoch": epoch,
            "Accuracy": acc,
            "Loss": loss,
            "Timestamp": time.time()
        })

    def print_dashboard(self):
        self.console.clear()

        latest = self.global_history[-1]
        prev = self.global_history[-2] if len(self.global_history) > 1 else latest
        acc_now = latest["Accuracy"]
        loss_now = latest["Loss"]

        window = 10  # Rolling window size
        total_epochs = len(self.global_history)

        # === Moving Avg Accuracy Delta ===
        if total_epochs >= window:
            acc_window = [entry["Accuracy"] for entry in self.global_history[-window:]]
            acc_change = acc_window[-1] - acc_window[0]
            avg_acc_delta = acc_change / (window - 1)
        else:
            acc_window = [entry["Accuracy"] for entry in self.global_history]
            acc_change = acc_window[-1] - acc_window[0]
            avg_acc_delta = acc_change / max(1, len(acc_window) - 1)

        # === Moving Avg Time per Epoch ===
        if total_epochs >= window:
            recent_times = [self.global_history[i]["Timestamp"] for i in range(-window, 0)]
            time_deltas = [recent_times[i+1] - recent_times[i] for i in range(window - 1)]
            avg_time = sum(time_deltas) / len(time_deltas)
        else:
            avg_time = (time.time() - self.start_time) / total_epochs


        global_table = Table(title="🌍 Global Training Summary")
        global_table.add_column("Epoch")
        global_table.add_column("Accuracy")
        global_table.add_column("Δ/epoch")
        global_table.add_column("Loss")
        global_table.add_column("Avg Time")

        global_table.add_row(
            str(latest["Epoch"]),
            f"{acc_now:.2%}",
            Text(f"{avg_acc_delta:+.2%}", style="green" if avg_acc_delta > 0 else "red"),
            f"{loss_now:.4f}",
            f"{avg_time:.2f}s"
        )

        self.console.print(global_table)

        # === Per Node Summary ===
        node_table = Table(title="🧩 Node Snapshots")
        for col in ["Node", "Start Epoch", "Start Acc", "Current Acc", "Δ Acc", "Temp",
                    "Gate Entropy", "ProtoDiv", "NodeDiv", "Recon"]:
            node_table.add_column(col)

        for node in self.node_snapshots:
            acc_start = node["Start Acc"]
            acc_now = node["Current Acc"]
            acc_delta = acc_now - acc_start
            acc_color = "green" if acc_delta > 0 else "red"

            node_table.add_row(
                str(node["Node"]),
                str(node["Start Epoch"]),
                f"{acc_start:.2%}",
                f"{acc_now:.2%}",
                Text(f"{acc_delta:+.2%}", style=acc_color),
                f"{node['Temp']:.4f}" if node['Temp'] else "-",
                f"{node['Gate Entropy']:.4f}" if node['Gate Entropy'] else "-",
                f"{node['ProtoDiv']:.7f}" if node['ProtoDiv'] else "-",
                f"{node.get('NodeDiv', '-'): .4f}" if node.get("NodeDiv") else "-",
                f"{node['Recon']:.4f}" if node['Recon'] else "-",
            )

        self.console.print(node_table)

    def final_summary(self, model, final_accuracy, total_duration):
        self.console.rule("[bold green]🏁 Final Training Summary")

        final_table = Table(show_header=True)
        final_table.add_column("Test Accuracy", justify="center")
        final_table.add_column("Total Duration", justify="center")
        final_table.add_column("Node Count", justify="center")

        total_minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        time_str = f"{total_minutes}m {seconds}s"

        final_table.add_row(
            f"{final_accuracy:.2%}",
            time_str,
            str(model.node_count)
        )

        self.console.print(final_table)
        self.console.rule()
