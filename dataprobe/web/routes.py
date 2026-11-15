"""页面路由与 JSON API。"""
from __future__ import annotations

from flask import (
    Flask, jsonify, redirect, render_template, request, url_for,
)

from .. import storage
from ..config import ST_FALSE_POSITIVE, THRESHOLD_SCHEMA
from ..infer import T_DATETIME, T_TEXT

METRIC_LABELS = {
    "row_volume": "行数波动", "null_rate": "空值率漂移",
    "numeric_distribution": "数值分布漂移(PSI/KS)",
    "category_distribution": "类别分布漂移(PSI)",
    "pattern_drift": "文本格式漂移", "range_shift": "取值范围偏移",
    "cardinality": "基数漂移", "bool_ratio": "布尔占比漂移",
    "column_added": "新增列", "column_dropped": "列消失",
    "column_renamed": "列改名", "column_reordered": "列移位",
    "type_changed": "类型变化",
}
SEVERITY_LABELS = {"info": "信息", "warning": "预警", "critical": "严重"}


def register(app: Flask, get_conn) -> None:

    # ---------------- 页面 ----------------

    @app.route("/")
    def index():
        conn = get_conn()
        tables = storage.list_all_tables(conn)
        open_critical = len(storage.list_events(conn, severity="critical",
                                                 status="open", limit=10000))
        open_warning = len(storage.list_events(conn, severity="warning",
                                               status="open", limit=10000))
        return render_template("index.html", tables=tables,
                               open_critical=open_critical,
                               open_warning=open_warning)

    @app.route("/sources")
    def sources_page():
        conn = get_conn()
        return render_template("sources.html", sources=storage.list_sources(conn))

    @app.route("/tables/<table_id>")
    def table_detail(table_id):
        conn = get_conn()
        source = next((s for s in storage.list_sources(conn)
                       if s["table_id"] == table_id), None)
        if not source:
            return redirect(url_for("index"))
        daily = storage.list_table_daily(conn, table_id)
        latest = daily[-1]["snapshot_date"] if daily else None
        latest_cols = storage.get_columns_on_date(conn, table_id, latest) if latest else []
        idents = {i["stable_col_id"]: i for i in storage.list_identities(conn, table_id)}
        recent_events = storage.list_events(conn, table_id=table_id, limit=15)
        return render_template("table_detail.html", source=source, daily=daily,
                               latest=latest, latest_cols=latest_cols,
                               idents=idents, recent_events=recent_events,
                               metric_labels=METRIC_LABELS,
                               severity_labels=SEVERITY_LABELS)

    @app.route("/tables/<table_id>/columns/<path:column_name>")
    def column_detail(table_id, column_name):
        conn = get_conn()
        date = request.args.get("date") or storage.get_latest_snapshot_date(conn, table_id)
        if not date:
            return redirect(url_for("table_detail", table_id=table_id))
        cur = storage.get_column_daily(conn, table_id, date, column_name)
        if not cur:
            return redirect(url_for("table_detail", table_id=table_id))
        history = storage.get_column_history_by_identity(
            conn, table_id, cur["stable_col_id"])
        idents = {i["stable_col_id"]: i for i in storage.list_identities(conn, table_id)}
        events = storage.list_events(conn, table_id=table_id, limit=200)
        events = [e for e in events if e["stable_col_id"] == cur["stable_col_id"]]
        return render_template("column_detail.html", table_id=table_id, cur=cur,
                               history=history, events=events,
                               ident=idents.get(cur["stable_col_id"]),
                               is_text=cur["inferred_type"] == T_TEXT,
                               is_datetime=cur["inferred_type"] == T_DATETIME,
                               metric_labels=METRIC_LABELS,
                               severity_labels=SEVERITY_LABELS)

    @app.route("/events")
    def events_page():
        conn = get_conn()
        filters = {
            "table_id": request.args.get("table_id") or None,
            "severity": request.args.get("severity") or None,
            "status": request.args.get("status") or None,
            "metric": request.args.get("metric") or None,
        }
        events = storage.list_events(conn, include_suppressed=True, limit=5000, **filters)
        sources = {s["table_id"]: s["name"] for s in storage.list_sources(conn)}
        feedback = storage.list_feedback(conn)
        return render_template("events.html", events=events, sources=sources,
                               filters=filters, metrics=sorted(METRIC_LABELS),
                               metric_labels=METRIC_LABELS,
                               severity_labels=SEVERITY_LABELS,
                               feedback=feedback)

    @app.route("/thresholds")
    def thresholds_page():
        conn = get_conn()
        return render_template("thresholds.html",
                               schema=THRESHOLD_SCHEMA,
                               defaults=storage.get_effective_thresholds(conn),
                               overrides=storage.list_overrides(conn),
                               sources=storage.list_sources(conn))

    # ---------------- 来源 API ----------------

    @app.post("/api/sources")
    def api_add_source():
        f = request.form
        storage.upsert_source(get_conn(), name=f["name"].strip(), kind=f["kind"],
                              path=f["path"].strip(),
                              table_name=(f.get("table_name") or "").strip() or None,
                              note=f.get("note", ""))
        return redirect(url_for("sources_page"))

    @app.post("/api/sources/<name>/delete")
    def api_delete_source(name):
        storage.delete_source(get_conn(), name)
        return redirect(url_for("sources_page"))

    # ---------------- 事件 API ----------------

    @app.post("/api/events/<int:event_id>/status")
    def api_event_status(event_id):
        status = request.form["status"]
        conn = get_conn()
        if status == ST_FALSE_POSITIVE:
            row = conn.execute("SELECT * FROM events WHERE event_id=?",
                               (event_id,)).fetchone()
            if row:
                storage.add_feedback(conn, signature=row["signature"],
                                     table_id=row["table_id"], metric=row["metric"],
                                     note=request.form.get("note", ""))
        storage.update_event_status(conn, event_id, status)
        return redirect(request.referrer or url_for("events_page"))

    @app.post("/api/feedback/<signature>/delete")
    def api_feedback_delete(signature):
        storage.delete_feedback(get_conn(), signature)
        return redirect(url_for("events_page"))

    # ---------------- 阈值 API ----------------

    @app.post("/api/thresholds")
    def api_set_threshold():
        conn = get_conn()
        scope = request.form["scope"]
        key = request.form["key"]
        value = request.form["value"]
        table_id = request.form.get("table_id") or None
        storage.set_override(conn, scope, key, value, table_id)
        return redirect(url_for("thresholds_page"))

    @app.post("/api/thresholds/delete")
    def api_delete_threshold():
        conn = get_conn()
        storage.delete_override(conn, request.form["scope"], request.form["key"],
                                request.form.get("table_id") or None)
        return redirect(url_for("thresholds_page"))

    # ---------------- 图表/数据 API ----------------

    @app.get("/api/tables/<table_id>/series")
    def api_table_series(table_id):
        conn = get_conn()
        daily = storage.list_table_daily(conn, table_id)
        return jsonify([{"date": d["snapshot_date"], "rows": d["row_count"],
                         "columns": d["column_count"], "weekday": d["weekday"]}
                        for d in daily])

    @app.get("/api/tables/<table_id>/columns/<path:column_name>/series")
    def api_column_series(table_id, column_name):
        conn = get_conn()
        date = storage.get_latest_snapshot_date(conn, table_id)
        cur = storage.get_column_daily(conn, table_id, date, column_name)
        if not cur:
            return jsonify({"error": "column not found"}), 404
        history = storage.get_column_history_by_identity(
            conn, table_id, cur["stable_col_id"])
        out = []
        for h in history:
            p = h["profile"]
            point = {"date": h["snapshot_date"], "null_rate": h["null_rate"],
                     "distinct_count": h["distinct_count"],
                     "distinct_ratio": h["distinct_ratio"],
                     "inferred_type": h["inferred_type"]}
            if p.get("numeric"):
                point["median"] = p["numeric"]["median"]
                point["mean"] = p["numeric"]["mean"]
                point["min"] = p["numeric"]["min"]
                point["max"] = p["numeric"]["max"]
            if p.get("boolean"):
                point["true_ratio"] = p["boolean"]["true_ratio"]
            out.append(point)
        return jsonify(out)
