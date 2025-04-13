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
            "Gate Entropy Init": None,
            "Gate Entropy Now": None,
            "ProtoDiv": None,
            "Recon": None,
            "Usage": None,
            "Variance": None,
            "Temp": None
        })

    def update_node_metrics(self, node_idx, acc, entropy, entropy_init, proto_div, recon, usage, variance, temp):
        node = self.node_snapshots[node_idx]
        node["Current Acc"] = acc
        node["Gate Entropy Now"] = entropy
        node["Gate Entropy Init"] = entropy_init if node["Gate Entropy Init"] is None else node["Gate Entropy Init"]
        node["ProtoDiv"] = proto_div
        node["Recon"] = recon
        node["Usage"] = usage
        node["Variance"] = variance
        node["Temp"] = temp

    def log_epoch(self, epoch, acc, loss):
        self.global_history.append({
            "Epoch": epoch,
            "Accuracy": acc,
            "Loss": loss,
        })

    def print_dashboard(self):
        self.console.clear()

        latest = self.global_history[-1]
        prev = self.global_history[-2] if len(self.global_history) > 1 else latest
        acc_now = latest["Accuracy"]
        acc_prev = prev["Accuracy"]
        loss_now = latest["Loss"]

        total_epochs = len(self.global_history)
        avg_acc_delta = (acc_now - self.global_history[0]["Accuracy"]) / max(1, total_epochs - 1)
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
                    "Gate Entropy Δ", "ProtoDiv", "Recon", "Usage", "Variance"]:
            node_table.add_column(col)

        for node in self.node_snapshots:
            acc_start = node["Start Acc"]
            acc_now = node["Current Acc"]
            acc_delta = acc_now - acc_start
            acc_color = "green" if acc_delta > 0 else "red"

            entropy_delta = None
            if node["Gate Entropy Now"] is not None and node["Gate Entropy Init"] is not None:
                entropy_delta = node["Gate Entropy Now"] - node["Gate Entropy Init"]
            entropy_color = "green" if entropy_delta and entropy_delta < 0 else "red"

            node_table.add_row(
                str(node["Node"]),
                str(node["Start Epoch"]),
                f"{acc_start:.2%}",
                f"{acc_now:.2%}",
                Text(f"{acc_delta:+.2%}", style=acc_color),
                f"{node['Temp']:.3f}" if node['Temp'] else "-",
                Text(f"{entropy_delta:+.2f}" if entropy_delta else "-", style=entropy_color if entropy_delta else ""),
                f"{node['ProtoDiv']:.4f}" if node['ProtoDiv'] else "-",
                f"{node['Recon']:.4f}" if node['Recon'] else "-",
                f"{node['Usage']:.2f}" if node['Usage'] else "-",
                f"{node['Variance']:.4f}" if node['Variance'] else "-"
            )

        self.console.print(node_table)

    def final_summary(self, model, final_accuracy, total_duration):
        self.console.rule("[bold green]🏁 Final Training Summary")

        final_table = Table(show_header=True)
        final_table.add_column("Final Accuracy", justify="center")
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
