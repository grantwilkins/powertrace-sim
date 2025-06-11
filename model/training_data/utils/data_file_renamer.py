import re
from datetime import datetime
from pathlib import Path

CLIENT_DIR = Path("/Users/grantwilkins/powertrace-sim/data/edgecases-llama-3-8b-h100")

json_re = re.compile(
    r"""^vllm-
        (?P<qps>[\d.]+)qps
        -tp(?P<tp>\d+)
        -(?P<raw_model>.+?)
        -(?P<json_dt>\d{8}-\d{6})
        \.json$
    """,
    re.VERBOSE,
)
model_map = {
    "Llama-3.1-8B-Instruct": "llama-3-8b",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-distill-8b",
    "Llama-3.1-70B-Instruct": "llama-3-70b",
    "DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-70b",
}

csv_re = re.compile(r"_d(?P<csv_dt>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv$")

for json_path in CLIENT_DIR.glob("vllm-*qps-tp*-*.json"):
    m = json_re.match(json_path.name)
    if not m:
        continue

    qps = m.group("qps")
    tp = m.group("tp")
    raw_model = m.group("raw_model")
    jdt_str = m.group("json_dt")

    json_dt = datetime.strptime(jdt_str, "%Y%m%d-%H%M%S")

    model_csv = model_map.get(
        raw_model, raw_model.lower().replace(".", "").replace("-instruct", "")
    )
    pattern = f"{model_csv}_tp{tp}_p{qps}_d*.csv"
    candidates = []
    for csv_path in CLIENT_DIR.glob(pattern):
        cm = csv_re.search(csv_path.name)
        if not cm:
            continue
        cdt_str = cm.group("csv_dt")
        csv_dt = datetime.strptime(cdt_str, "%Y-%m-%d-%H-%M-%S")
        delta = (json_dt - csv_dt).total_seconds()
        if delta > 0:
            candidates.append((delta, csv_path, csv_dt))

    if not candidates:
        print(f"No CSV found for {json_path.name}")
        continue

    _, best_csv, _ = min(candidates, key=lambda x: x[0])
    new_dt_str = json_dt.strftime("%Y%m%d-%H%M%S")
    new_name = f"{model_csv}_tp{tp}_p{qps}_d{new_dt_str}.csv"

    best_csv.rename(best_csv.with_name(new_name))
    print(f"{best_csv.name} → {new_name}")
