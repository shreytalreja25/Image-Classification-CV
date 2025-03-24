import os
import re
from pathlib import Path
from collections import defaultdict

def extract_metrics_from_report(report_path):
    model_name = "Unknown"
    accuracy = macro_f1 = "N/A"
    timestamp = report_path.parent.name.split("_eval_")[-1]

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    model_match = re.search(r"Model:\s*(.+)", content)
    if model_match:
        model_name = model_match.group(1).strip()

    acc_match = re.search(r"accuracy\s+([\d.]+)", content)
    if acc_match:
        accuracy = f"{float(acc_match.group(1)) * 100:.2f}%"

    f1_match = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content)
    if f1_match:
        macro_f1 = f"{float(f1_match.group(1)) * 100:.2f}%"

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Timestamp": timestamp,
        "Path": report_path
    }

def find_reports(root_dir):
    reports = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "report.txt":
                reports.append(Path(subdir) / file)
    return reports

def get_latest_reports(reports):
    latest = {}
    for report in reports:
        info = extract_metrics_from_report(report)
        model = info["Model"]
        timestamp = info["Timestamp"]
        if model not in latest or timestamp > latest[model]["Timestamp"]:
            latest[model] = info
    return list(latest.values())

def print_table(rows):
    print("\n📊 Latest Model Evaluation Summary:\n")
    headers = ["Model", "Accuracy", "Macro F1", "Timestamp"]
    col_widths = [max(len(str(row[h])) for row in rows + [dict(zip(headers, headers))]) for h in headers]

    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("-|-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths)))

def save_summary(rows, path="results/summary_report.txt"):
    with open(path, "w", encoding="utf-8") as f:
        headers = ["Model", "Accuracy", "Macro F1", "Timestamp"]
        col_widths = [max(len(str(row[h])) for row in rows + [dict(zip(headers, headers))]) for h in headers]

        f.write(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + "\n")
        f.write("-|-".join("-" * w for w in col_widths) + "\n")
        for row in rows:
            f.write(" | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths)) + "\n")

    print(f"\n✅ Summary saved to: {path}")

def main():
    all_reports = find_reports("results/ML_results") + find_reports("results/DL_results")
    if not all_reports:
        print("❌ No evaluation reports found.")
        return

    latest_results = get_latest_reports(all_reports)
    latest_results = sorted(latest_results, key=lambda x: x["Timestamp"])
    print_table(latest_results)
    save_summary(latest_results)

if __name__ == "__main__":
    main()
