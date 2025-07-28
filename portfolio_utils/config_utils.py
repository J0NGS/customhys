import os
import json
from datetime import datetime
import pandas as pd

def load_hh_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Arquivo de configuração {config_file} não encontrado")
    with open(config_file, 'r') as f:
        lines = []
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                lines.append(line)
        config_text = '\n'.join(lines)
        try:
            config = json.loads(config_text)
            return config
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON do arquivo {config_file}: {e}")
            return None

def create_output_directory(instance_file, initial_scheme):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    dir_name = f"{timestamp}_{instance_name}_{initial_scheme}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def _save_json(data, path, default=None):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=default)

def _build_instance_summary(instance_data):
    return {
        "n_assets": instance_data["n_assets"],
        "returns": instance_data["returns"].tolist(),
        "std_devs": instance_data["std_devs"].tolist(),
        "cov_matrix": instance_data["cov_matrix"].tolist()
    }

def _build_stats(execution_logs):
    best_solution = min(execution_logs, key=lambda x: x.get('objective', float('inf')))
    return {
        "total_evaluations": len(execution_logs),
        "best_objective": best_solution.get('objective'),
        "best_sharpe": best_solution.get('sharpe'),
        "best_return": best_solution.get('expected_return'),
        "best_risk": best_solution.get('risk'),
        "best_weights": best_solution.get('weights'),
        "best_selected_assets": best_solution.get('selected_assets')
    }

def save_logs(output_dir, execution_logs, instance_data, hh_config, result):
    _save_json(hh_config, os.path.join(output_dir, "hh_config.json"))
    instance_summary = _build_instance_summary(instance_data)
    _save_json(instance_summary, os.path.join(output_dir, "instance_data.json"))
    _save_json(result, os.path.join(output_dir, "hh_result.json"), default=str)
    if execution_logs:
        stats = _build_stats(execution_logs)
        _save_json(stats, os.path.join(output_dir, "summary_stats.json"))
