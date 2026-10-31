"""Rebuild the album's local assets from committed history, CSVs and notebook outputs.

Run from any directory with: python scripts/build_album.py
Only Python's standard library is required. No experiments are retrained.
"""

import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "assets" / "album"
# Freeze the pre-cleanup snapshot, so future rebuilds do not count maintenance code.
ARCHIVE_REF = "35628983c967974a8943e4bb4e300e0d5915c561"


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT).decode("utf-8").strip()


def build():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    photos = []
    selections = [
        ("notebook/features-selection-learn.ipynb", 23, 0, "roc", "一条曲线，衡量一次判断", "GBDT 的 ROC 曲线，来自特征选择学习笔记。"),
        ("notebook/features-selection-learn.ipynb", 84, 0, "lgb-ifs", "特征不是越多越好", "LightGBM 的增量特征曲线，保留当时 Notebook 中的输出。"),
        ("notebook/features-selection-learn.ipynb", 133, 0, "rf-ifs", "在森林里寻找线索", "随机森林的特征数量与 MCC 记录。"),
        ("notebook/im-learn.ipynb", 16, 0, "sampling", "让少数样本，也被看见", "不平衡学习笔记中的二维可视化。"),
    ]
    for notebook, cell, output, slug, title, caption in selections:
        document = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
        encoded = document["cells"][cell]["outputs"][output]["data"]["image/png"]
        payload = base64.b64decode("".join(encoded))
        filename = f"{slug}.png"
        (DESTINATION / filename).write_bytes(payload)
        photos.append({"image": f"assets/album/{filename}", "title": title, "caption": caption,
                       "source": notebook, "cell": cell + 1, "sha256": hashlib.sha256(payload).hexdigest()})

    configurations = []
    for mcc_path in sorted((ROOT / "logs").glob("mcc-sum-*.csv")):
        key = mcc_path.stem.removeprefix("mcc-sum-")
        values = {}
        for metric in ("mcc", "acc"):
            path = ROOT / "logs" / f"{metric}-sum-{key}.csv"
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
            if not rows or any(row[0] != f"f{i}" for i, row in enumerate(rows, 1)):
                raise ValueError(f"Unexpected feature sequence: {path}")
            values[metric] = [float(row[1]) for row in rows]
            if not all(-1 <= value <= 1 for value in values[metric]):
                raise ValueError(f"Out-of-range metric: {path}")
        if len(values["mcc"]) != len(values["acc"]):
            raise ValueError(f"Mismatched metric lengths: {key}")
        match = re.fullmatch(r"c(\d+)-l(\d+)-u(\d+)-e(\d+)-lr(.*)-d(.*)", key)
        if match is None:
            raise ValueError(f"Unrecognized log name: {key}")
        classes, layers, units, epochs, rate, dropout = match.groups()
        configurations.append({"id": key, "label": f"{layers} 层 · {units} 单元 · LR {rate} · {epochs} epochs",
            "layers": int(layers), "units": int(units), "epochs": int(epochs), "rate": rate,
            "dropoutCode": dropout, "mcc": values["mcc"], "acc": values["acc"],
            "source": {metric: f"logs/{metric}-sum-{key}.csv" for metric in values}})

    committed = git("ls-tree", "-r", "--name-only", ARCHIVE_REF).splitlines()
    # New maintenance files do not change the historical archive count.
    historical = [name for name in committed if not name.startswith(("experiments/", "tests/"))
                  and name not in {"scripts/build_album.py", "scripts/review_archive.py"}]
    timeline = []
    milestones = [
        ("2017", "从一个分类器开始", "NumPy、scikit-learn、TensorFlow；从基础练习走向 SNN、CNN 与交叉验证。", "Initial commit"),
        ("2018", "把序列交给网络", "RNN、肽数据与胶囊网络，模型结构开始回应具体的数据问题。", "capsNet/peptide_capsnet.py"),
        ("2019", "让比较变得更丰富", "PyTorch FNN、BN / LN、传统分类器和多标签实验，留下更多可以比较的路径。", "nn/fnn.py"),
        ("2020", "追问：哪些特征更重要？", "增量特征选择、分类器比较与 Stacking，把注意力从单个模型移向实验流程。", "sci-fs/clf_fs.py"),
        ("2023", "后来，探索仍有续页", "符号回归与 SVR 的补充记录，成为这份研究生工作档案之后的延伸。", "gp-symbolic-regressor.py"),
    ]
    for year, title, description, source in milestones:
        if source == "Initial commit":
            line = git("log", ARCHIVE_REF, "--reverse", "--format=%ad|%h|%s", "--date=short").splitlines()[0]
            path = "README.md"
        else:
            line = git("log", ARCHIVE_REF, "--reverse", "--format=%ad|%h|%s", "--date=short", "--", source).splitlines()[0]
            path = source
        date, commit, message = line.split("|", 2)
        timeline.append({"year": year, "title": title, "description": description, "date": date,
                         "commit": commit, "message": message, "source": path})
    data = {"title": "那些年，与模型一起走过的日子", "photos": photos, "configurations": configurations,
            "timeline": timeline, "counts": {"python": sum(p.endswith(".py") for p in historical),
                "notebooks": sum(p.endswith(".ipynb") for p in historical),
                "logs": len(configurations) * 2, "features": len(configurations[0]["mcc"])},
            "provenance": "图像直接提取自 Notebook；曲线逐点读取 logs/；时间线日期来自 Git 提交。历史指标未重新验证。"}
    (DESTINATION / "data.js").write_text("window.ALBUM_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    # A shareable scientific figure, also displayed if JavaScript is disabled.
    chart = configurations[0]
    width, height, left, top, graph_w, graph_h = 1000, 480, 72, 45, 880, 345
    points = " ".join(f"{left + i * graph_w / (len(chart['mcc']) - 1):.2f},{top + (1 - value) * graph_h:.2f}"
                      for i, value in enumerate(chart["mcc"]))
    grid = "".join(f'<line x1="72" x2="952" y1="{top + i * graph_h / 5}" y2="{top + i * graph_h / 5}" stroke="#deded5"/>'
                   f'<text x="57" y="{top + i * graph_h / 5 + 5}" text-anchor="end">{1-i/5:.1f}</text>' for i in range(6))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">Historical SNN incremental feature curve</title><desc id="desc">941 MCC values from {html.escape(chart["source"]["mcc"])}</desc><rect width="1000" height="480" fill="#f7f4ed"/><g font-family="sans-serif" font-size="14" fill="#54635c">{grid}<polyline points="{points}" fill="none" stroke="#3d6a59" stroke-width="1.2"/><text x="72" y="25">MCC / SNN / {chart["layers"]} layers / LR {chart["rate"]}</text><text x="72" y="420">1</text><text x="952" y="420" text-anchor="end">941 features</text><text x="72" y="460" font-size="12">Historical log; exploratory values, not a newly validated benchmark.</text></g></svg>'
    (DESTINATION / "historical-ifs.svg").write_text(svg, encoding="utf-8")
    (DESTINATION / "sources.json").write_text(json.dumps({"photos": photos, "timeline": timeline,
        "counts": data["counts"], "archive_ref": ARCHIVE_REF,
        "log_sources": [c["source"] for c in configurations]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Album assets: {len(photos)} original notebook images, {len(configurations)} paired curves; {data['counts']}")


if __name__ == "__main__":
    build()
